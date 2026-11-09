"""First-token probing and stream reassembly."""

import json

from inference import sse


def chunk(**delta):
    return ('data: ' + json.dumps({
        "id": "chatcmpl-1", "model": "m",
        "choices": [{"index": 0, "delta": delta}]}) + "\n\n").encode()


def test_reasoning_moves_ttft_but_not_ttfc():
    probe = sse.FirstTokenProbe()
    probe.feed(chunk(reasoning_content="thinking"), 100.0)
    assert probe.ttft_ms == 100.0
    assert probe.ttfc_ms is None, "reasoning is not something a user can see"
    probe.feed(chunk(content="answer"), 900.0)
    assert probe.ttft_ms == 100.0
    assert probe.ttfc_ms == 900.0
    # The gap is the deferral cost: 800ms of a user watching nothing happen.
    assert probe.ttfc_ms - probe.ttft_ms == 800.0


def test_a_tool_call_counts_as_visible():
    probe = sse.FirstTokenProbe()
    probe.feed(chunk(tool_calls=[{"index": 0}]), 50.0)
    assert probe.ttfc_ms == 50.0


def test_keys_split_across_chunks_are_still_found():
    probe = sse.FirstTokenProbe()
    raw = chunk(content="hello")
    probe.feed(raw[:30], 10.0)
    probe.feed(raw[30:], 20.0)
    assert probe.ttfc_ms is not None, "SCAN_OVERLAP should bridge the boundary"


def test_reassemble_rebuilds_the_message():
    body = chunk(content="he") + chunk(content="llo") + b"data: [DONE]\n\n"
    result = sse.reassemble(body)
    assert result["choices"][0]["message"]["content"] == "hello"
    assert result["_reassembled_from_stream"] is True


def test_reassemble_carries_engine_specific_blocks():
    # MTPLX puts its whole per-request record on the final chunk. Dropping it
    # would mean streamed turns - which is every real client - lost the decode
    # rate, peak memory and speculative-decoding acceptance.
    final = ('data: ' + json.dumps({
        "id": "chatcmpl-1",
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        "usage": {"completion_tokens": 2},
        "timings": {"predicted_per_second": 59.4},
        "mtplx_stats": {"mtp_depth": 3, "peak_memory_bytes": 1},
    }) + "\n\n").encode()
    result = sse.reassemble(chunk(content="hi") + final)
    assert result["id"] == "chatcmpl-1"
    assert result["timings"]["predicted_per_second"] == 59.4
    assert result["mtplx_stats"]["mtp_depth"] == 3


def test_tool_call_arguments_are_concatenated_across_chunks():
    body = (chunk(tool_calls=[{"index": 0, "id": "c1", "type": "function",
                               "function": {"name": "read_file",
                                            "arguments": '{"pa'}}])
            + chunk(tool_calls=[{"index": 0,
                                 "function": {"arguments": 'th":"a"}'}}]))
    call = sse.reassemble(body)["choices"][0]["message"]["tool_calls"][0]
    assert call["function"]["name"] == "read_file"
    assert json.loads(call["function"]["arguments"]) == {"path": "a"}
