# 2026-10-16 rescue backup (servers about to be reinstalled)

Work rescued before the reinstall. The tunnel out of the office PC runs at
about 20-50 KB/s, so repo content was pushed to GitHub from the servers
themselves; only the machines without any GitHub route (234, 247) had their
working-tree delta streamed straight back to this PC.

## Recovery branches (all in Coding-Pangolin/flash-linear-attention-npu)

| branch | source machine | content |
|---|---|---|
| backup/2026-10-16-host241 | 241 /home/npu_user7/BartonFang/flash-linear-attention-npu | feat/ppfm-tile-a5 (17eab762) plus the untracked chunk_kda_fwd_prepare design docs |
| backup/2026-10-16-host241-race | 241 ~/BartonFang/ppfm-race/repo | race sandbox branch (95cb364a), pre_process_fwd_kernel_merged.cpp changes, gate scripts and docs |
| backup/2026-10-16-host246 | 246 /data/admin123/bartonfang/flash-linear-attention-npu | feat/ppfm-tile-a5 (879857f6) plus ppfm docs (H20 baseline, PROFILING_GUIDE, interface_vs_competitor) |
| backup/2026-10-16-host246-home | 246 /home/admin123/flash-linear-attention-npu | main (e22e0bf4) plus the whole pre_process_fwd_kernel_merged operator dir (was untracked) |

## What is in this directory

* 234_192.168.9.234/kda_a2_scripts.tgz - the whole 234 experiment harness:
  acc_ab.py / acc_cmp.py / bench_*.sh / probe*.py and friends from
  /home/admin123/kda_a2 (451 files, 63.8 MB unpacked, 36 MB packed) plus
  src_snap/ and out/ logs. This tree exists nowhere else, so it is the single
  most valuable thing in this branch.
* 234_192.168.9.234/wip_delta.tgz - the 7 modified files of kda_a2/repo plus the
  untracked *.a2v_snapshot files and chunk_kda_fwd_prepare/docs/, paths relative
  to the repo root. Unpack with: tar xzf wip_delta.tgz -C <repo_root>
* 234_192.168.9.234/wip_meta.txt - branch, HEAD, git status and full git diff
  of that repo, for cross-checking.
* 247_192.168.13.247/wip_delta.tgz - fla/ops/ascendc/cp/ of the 247 repo
  (35 changed entries), paths relative to the repo root.
* 247_192.168.13.247/wip_meta.txt - same metadata plus the full diff.
* _gitstate/*.txt - branch/HEAD/status/diffstat snapshot of every repo we
  touched, taken before the servers go down.

## How to restore

    git fetch <fork> 'refs/heads/backup/*:refs/remotes/fork/backup/*'
    git checkout feat/ppfm-tile-a5
    tar xzf wip_delta.tgz -C .        # put new files back into the worktree
    git apply wip_meta.txt            # or diff by hand from the metadata

## Deliberately NOT rescued (will be lost)

* 234 /home/admin123/kda_a2/cache/*.pt (~11 GB accuracy reference caches),
  acc/ (~7.2 GB per-iteration accuracy runs) and extra-info/data-dump
  (4.8 GB exception dumps). All are regenerable and far too large for the link.
* build/ trees, venvs/conda envs and CANN installers on every machine.
* 247 /data/admin123/flash-linear-attention-npu/tests (1.5 GB ATK data).

## Security note

The git remotes on 246 and 247 embed a GitHub PAT in plain text (ghp_...).
Those machines were just compromised, so revoke that token on GitHub and issue
a new one.
