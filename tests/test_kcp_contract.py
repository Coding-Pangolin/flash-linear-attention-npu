"""Independent numerical, scheduling and measurement checks for KDA CP."""
import importlib.util
from pathlib import Path
import random
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]


def load(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


exp = load("kcp_experiment", "scripts/kcp_experiment.py")
ref = load("kcp_reference", "tests/atk/chunk_kda_cp_fwd/reference.py")
packed = load("kcp_packed", "torch_custom/fla_npu/fla_npu/ops/ascendc/_kcp_packed.py")
kcp = load("kcp_dispatch", "torch_custom/fla_npu/fla_npu/ops/ascendc/_kcp.py")


class KcpContractTests(unittest.TestCase):
    def test_chunk32_zero_round_selection_is_group_wide(self):
        allowed = packed.build_packed_plan(64, 1, 2, 27, cu_seqlens=[*range(65), 128])
        fallback = packed.build_packed_plan(65, 1, 2, 27, cu_seqlens=[*range(66), 130])
        crossing = packed.build_packed_plan(64, 1, 2, 27)
        many_heads = packed.build_packed_plan(128, 128, 2, 27, cu_seqlens=[0, 128, 256])
        self.assertTrue(kcp._uses_chunk32_zero_round(allowed))
        self.assertFalse(kcp._uses_chunk32_zero_round(fallback))
        self.assertFalse(kcp._uses_chunk32_zero_round(crossing))
        self.assertFalse(kcp._uses_chunk32_zero_round(many_heads))

    def test_packed_boundary_roles_and_workspace(self):
        plan = packed.build_packed_plan(10, 32, 4, 27, cu_seqlens=[0, 13, 17, 28, 40])
        self.assertEqual(tuple(x.cu_seqlens for x in plan.shards),
                         ((0, 10), (0, 3, 7, 10), (0, 8, 10), (0, 10)))
        self.assertEqual(tuple(x.prefix_start_rank for x in plan.shards), (0, 0, 1, 2))
        self.assertEqual(tuple(x.sends_summary for x in plan.shards), (True, True, True, False))
        self.assertEqual((plan.head_group_size, plan.rounds, plan.batch_slots, plan.receive_slots),
                         (16, 2, 2, 2))
        owners = {plan.owner(seq, h, 1) for seq in range(3) for h in range(32)}
        self.assertEqual(owners, set(range(27)))
        for world, mib in ((2, 34.3125), (4, 42.3125), (8, 58.3125)):
            whole = packed.build_packed_plan(8192, 32, world, 27, head_group_size=32)
            split = packed.build_packed_plan(16384, 32, world, 27, head_group_size=16)
            self.assertEqual(whole.workspace(0)[1], mib * 2**20)
            self.assertEqual(split.workspace(0)[1], whole.workspace(0)[1])
            for offset, size in split.workspace(0)[0].values():
                self.assertEqual(offset % 65536, 0)
                self.assertEqual(size % 65536, 0)

    def test_packed_auto_plan_and_input_errors(self):
        import torch
        for world in (2, 4, 8):
            aligned = packed.build_packed_plan(10, 32, world, 27,
                                               cu_seqlens=torch.arange(world + 1) * 10)
            self.assertEqual((aligned.rounds, aligned.batch_slots, aligned.receive_slots), (0, 0, 0))
            self.assertEqual(aligned.workspace(0)[0]["receive"][1], 0)
            packed_starts = [0] + list(range(5, world * 10, 10)) + [world * 10]
            independent = packed.build_packed_plan(10, 32, world, 27, cu_seqlens=packed_starts)
            self.assertEqual(independent.rounds, 1)
            for heads in (15, 64):
                self.assertEqual(packed.build_packed_plan(10, heads, world, 27).rounds, 1)
            self.assertEqual(packed.build_packed_plan(10, 5, world, 27,
                                                      head_group_size=3).rounds, 2)
        for bad in ([1, 20], [0, 4, 4, 20], [0, 12, 9, 20], [0, 19], [False, 20],
                    [0., 20.], torch.tensor([[0, 20]]), torch.tensor([0, 20], device="meta"),
                    torch.tensor([0, 20], dtype=torch.int16)):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                packed.build_packed_plan(10, 32, 2, 27, cu_seqlens=bad)
        with self.assertRaises(ValueError):
            packed.build_packed_plan(10, 32, 2, 27, head_group_size=8)

    def test_packed_rank_mapping_on_192_boundary_sets(self):
        generator = random.Random(912)
        tokens = 10
        for world in (2, 4, 8):
            total = world * tokens
            for _ in range(64):
                cuts = generator.sample(range(1, total), generator.randrange(total))
                cu = [0, *sorted(cuts), total]
                plan = packed.build_packed_plan(tokens, 32, world, 27, cu_seqlens=cu)
                self.assertEqual(plan.rounds == 0, all(cut in cu for cut in range(tokens, total, tokens)))
                for rank, shard in enumerate(plan.shards):
                    start, end = rank * tokens, (rank + 1) * tokens
                    local = [(i, a, b) for i, (a, b) in enumerate(zip(cu, cu[1:])) if a < end and b > start]
                    self.assertEqual(shard.first_sequence, local[0][0])
                    self.assertEqual(shard.sends_summary, local[-1][-1] > end)
                    self.assertEqual(shard.cu_seqlens, (0, *(min(end, b) - start for _, _, b in local)))
                    first_rank = local[0][1] // tokens
                    self.assertEqual(shard.prefix_start_rank, first_rank)
                    for src in range(first_rank, rank):
                        previous = plan.shards[src]
                        self.assertTrue(previous.sends_summary)
                        self.assertEqual(previous.first_sequence + previous.sequences - 1,
                                         shard.first_sequence)

    def test_packed_oracle_keeps_one_state_per_local_segment(self):
        import torch
        generator = torch.Generator().manual_seed(32)
        q, k, v, g = (torch.randn(1, 1, 10, 128, generator=generator) for _ in range(4))
        beta = torch.randn(1, 1, 10, generator=generator)
        a = torch.zeros(1)
        args = (q, k, v, g, beta)
        first, first_state = ref.reference(*(x[:, :, :6] for x in args), A_log=a)
        second, mid_state = ref.reference(*(x[:, :, 6:8] for x in args), A_log=a)
        result, states = ref.packed_shard_reference(*args, A_log=a, cu_seqlens=[0, 6, 10], start=5, end=8)
        torch.testing.assert_close(result, torch.cat((first[:, 5:], second), dim=1), rtol=0, atol=0)
        torch.testing.assert_close(states, torch.cat((first_state, mid_state)), rtol=0, atol=0)

    def test_prefix_matrix_orientation_and_composition_order(self):
        import torch
        a = torch.zeros(4, 1, 128, 256)
        a[:, 0, :, 128:] = torch.eye(128)
        a[0, 0, :2, :2] = torch.tensor([[1., 2.], [3., 4.]])
        a[1, 0, :2, 128:130] = torch.tensor([[1., 1.], [0., 1.]])
        a[2, 0, :2, 128:130] = torch.tensor([[1., 0.], [1., 1.]])
        torch.testing.assert_close(ref.merge_reference(a, 2)[0, :2, :2],
                                   torch.tensor([[4., 6.], [3., 4.]], dtype=torch.float64))
        torch.testing.assert_close(ref.merge_reference(a, 3)[0, :2, :2],
                                   torch.tensor([[4., 6.], [7., 10.]], dtype=torch.float64))

    def test_ordered_prefix_is_exclusive_and_noncommutative(self):
        import torch
        a = torch.zeros(3, 1, 128, 256)
        a[:, 0, :, 128:] = torch.eye(128)
        a[0, ..., :128] = 2
        a[1, ..., :128] = 3
        a[1, ..., 128:] *= 4
        self.assertTrue(torch.equal(ref.merge_reference(a, 0), torch.zeros(1, 128, 128)))
        self.assertTrue(torch.all(ref.merge_reference(a, 1) == 2))
        self.assertTrue(torch.all(ref.merge_reference(a, 2) == 11))
        self.assertTrue(torch.all(ref.merge_reference(a[[1, 0, 2]], 2) == 5))

    def test_shard_continuation_and_gradients_equal_full_recurrence(self):
        import torch
        torch.manual_seed(7)
        q, k, v, g = [torch.randn(1, 1, 5, 128, requires_grad=True) for _ in range(4)]
        beta = torch.randn(1, 1, 5, requires_grad=True)
        a = torch.zeros(1, requires_grad=True)
        full, state = ref.reference(q, k, v, g, beta, A_log=a)
        front, h0 = ref.reference(q[:, :, :2], k[:, :, :2], v[:, :, :2],
                                 g[:, :, :2], beta[:, :, :2], A_log=a)
        back, final = ref.reference(q[:, :, 2:], k[:, :, 2:], v[:, :, 2:],
                                   g[:, :, 2:], beta[:, :, 2:], A_log=a, initial_state=h0)
        split = torch.cat((front, back), dim=1)
        torch.testing.assert_close(split, full, rtol=0, atol=0)
        torch.testing.assert_close(final, state, rtol=0, atol=0)
        leaves = (q, k, v, g, beta, a)
        for left, right in zip(torch.autograd.grad(full.sum(), leaves, retain_graph=True),
                               torch.autograd.grad(split.sum(), leaves)):
            torch.testing.assert_close(left, right, rtol=1e-6, atol=1e-7)

    def test_pipeline_all_group_counts_and_tails_drain(self):
        for heads in (1, 3, 8, 17, 64):
            for group in (1, 2, 4, 8, 16):
                steps = exp.schedule(heads, group)
                count = (heads + group - 1) // group
                self.assertEqual(sum(a == "consume" for a, _ in steps), count)
                self.assertEqual(steps[-1], ("consume", count - 1))

    def test_memory_includes_materialized_consumer_and_communication(self):
        result = exp.memory_plan(8192, 32, 8, 4, communication_bytes=32 * 2**20)
        self.assertEqual(result["regions"]["live_prepare"], 104 * 2**20)
        self.assertGreater(result["regions"]["live_consumer"], 0)
        self.assertFalse(exp.memory_plan(16384, 64, 8, 4)["within_temporary_budget"])
        self.assertGreater(exp.memory_plan(64, 8, 2, 4, training=True)["estimated_cache_bytes"], 0)
        saved = exp.memory_plan(16384, 64, 8, 4, training=True,
                                runtime_bytes=32*2**20, communication_bytes=64*2**20,
                                output_final_state=True)
        self.assertTrue(saved["within_temporary_budget"])
        self.assertEqual(saved["regions"]["live_consumer"], 0)
        self.assertEqual(saved["regions"]["h0"], 0)
        self.assertEqual(saved["regions"]["live_prepare"], 64 * 2**20)
        self.assertFalse(exp.memory_plan(16384, 64, 8, 4,
                                        runtime_bytes=32*2**20, communication_bytes=64*2**20)["within_temporary_budget"])

    def test_statistics_keep_slowest_rank_and_reject_incomplete_runs(self):
        rows = [{"window": w, "rank": r, "iteration": i, "duration_us": 10 + 10*r + w}
                for w in range(2) for r in range(2) for i in range(3)]
        self.assertEqual(exp.aggregate_windows(rows)["median_max_rank_mean_us"], 20.5)
        with self.assertRaises(ValueError):
            exp.aggregate_windows(rows[:-1])
        with self.assertRaises(ValueError):
            exp.aggregate_windows(rows + [rows[0]])
        with self.assertRaises(ValueError):
            exp.aggregate_windows([row for row in rows if row["iteration"] != 1])
        with self.assertRaises(ValueError):
            exp.aggregate_windows(rows, expected_ranks=4)
        combined = [dict(row, backend=b) for b in ("staged", "pipeline") for row in rows]
        self.assertEqual(set(exp.aggregate_windows(combined)["backends"]), {"staged", "pipeline"})


if __name__ == "__main__":
    unittest.main()
