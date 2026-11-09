"""Token rates: which measurement, and pooled rather than averaged."""

from inference import usage


SERVER = {
    "usage": {"prompt_tokens": 100, "completion_tokens": 50},
    "timings": {"prompt_n": 100, "predicted_n": 50, "prompt_ms": 500.0,
                "predicted_ms": 1000.0, "prompt_per_second": 200.0,
                "predicted_per_second": 50.0},
}
NO_TIMINGS = {"usage": {"prompt_tokens": 100, "completion_tokens": 50}}


def test_server_timings_are_preferred():
    figures = usage.rates(SERVER, ttft_ms=600.0, wall_ms=2000.0)
    assert figures["source"] == "server"
    # The server's own decode timer, not wall-minus-ttft (which would be 35.7).
    assert figures["decode_tps"] == 50.0
    assert figures["decode_ms"] == 1000.0


def test_client_arithmetic_when_the_server_says_nothing():
    figures = usage.rates(NO_TIMINGS, ttft_ms=600.0, wall_ms=2000.0)
    assert figures["source"] == "client"
    # 50 tokens over the 1400ms after the first one appeared.
    assert figures["decode_tps"] == 35.7
    assert figures["decode_ms"] == 1400.0


def test_the_two_sources_disagree_which_is_why_source_is_recorded():
    server = usage.rates(SERVER, ttft_ms=600.0, wall_ms=2000.0)
    client = usage.rates(NO_TIMINGS, ttft_ms=600.0, wall_ms=2000.0)
    assert server["decode_tps"] != client["decode_tps"]


def test_mtplx_stats_are_read_defensively():
    assert usage.mtplx_extras({}) == {}
    extras = usage.mtplx_extras({"mtplx_stats": {
        "drafted_by_depth": [3, 3, 3], "accepted_by_depth": [3, 2, 1],
        "peak_memory_bytes": 25_000_000_000, "ttft_s": 0.25,
        "mtp_depth": 3, "bonus_tokens": 4, "session_cache_hit": True,
    }})
    assert extras["mtp_accept"] == 0.667          # 6 accepted of 9 drafted
    assert extras["ttft_ms_server"] == 250.0
    assert extras["peak_rss_mb"] == 23841.9
    assert extras["cache_hit"] == 1


def test_missing_mtplx_fields_do_not_raise():
    extras = usage.mtplx_extras({"mtplx_stats": {"mtp_depth": 3}})
    assert extras["mtp_accept"] is None
    assert extras["peak_rss_mb"] is None
