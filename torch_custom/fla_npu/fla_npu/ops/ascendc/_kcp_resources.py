"""Pure CPU static accounting for materialized KDA CP buffers."""

LIMIT = 200_000_000


def memory_plan(tokens, heads, cp, group, *, training=False,
                runtime_bytes=0, communication_bytes=0, output_final_state=False,
                has_bias=True, external_beta_sigmoid=False):
    if any(type(x) is not int or x <= 0 for x in (tokens, heads, cp, group)):
        raise ValueError("tokens, heads, cp and group must be positive integers")
    if any(type(x) is not int or x < 0 for x in (runtime_bytes, communication_bytes)):
        raise ValueError("resource reservations must be nonnegative integer bytes")
    group = min(group, heads)
    chunks = (tokens + 63) // 64
    live_heads = min(heads, 2 * group)
    # Save-mode gk/Aqk/Akk/w/qg/kg/norm auxiliaries/H/V_new/h0 are required
    # returned cache storage. Only u and qg_scaled are transient Prepare data.
    # Inference materializes all six forward Prepare outputs and H/V_new.
    prepare_per_head = tokens * (4 * 128 if training else 12 * 128 + 2 * 64)
    consumer = 0 if training else chunks * group * 128 * 128 * 2 + tokens * group * 128 * 2
    parameter_bytes = 4 + (128 * 4 if has_bias else 0)
    multiple_groups = group < heads
    regions = {
        "live_prepare": live_heads * prepare_per_head,
        "live_consumer": consumer,
        "local_output_copy": tokens * group * 128 * 2 if multiple_groups else 0,
        "local_final_state_copy": group * 128 * 128 * 4 if multiple_groups and output_final_state else 0,
        "live_summaries": live_heads * 128 * 256 * 4 if cp > 1 else 0,
        "live_gather": cp * live_heads * 128 * 256 * 4 if cp > 1 else 0,
        "h0": 0 if training else group * 128 * 128 * 4,
        "parameter_copies": 0 if training else live_heads * parameter_bytes,
        "external_beta_sigmoid": tokens * group * 4 if external_beta_sigmoid else 0,
        "runtime_reservation": runtime_bytes,
        "communication_reservation": communication_bytes,
    }
    # Count only newly allocated cache storage; retained v/g/beta input views
    # are not duplicate allocations. Never count these outputs as scratch too.
    cache = (tokens * heads * (14 * 128 + 4 * 64 + 12)
             + chunks * heads * 128 * 128 * 2
             + tokens * heads * 128 * 2 + heads * 128 * 128 * 4
             + heads * parameter_bytes) if training else 0
    return {"regions": regions, "estimated_temporary_bytes": sum(regions.values()),
            "estimated_cache_bytes": cache, "within_temporary_budget": sum(regions.values()) <= LIMIT,
            "required_result_bytes": tokens * heads * 128 * 2 + (heads * 128 * 128 * 4 if output_final_state else 0),
            "requires_measured_peak": True}
