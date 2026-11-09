"""Parsing and rewriting the client's system prompt.

The fixtures here mirror the real VS Code shape from traces/turn-1-request.json:
an untagged preamble, a run of tagged blocks, and a *second* <instructions>
block at the end that wraps the workspace's skills and agents. That duplicate is
the whole reason rewriting is first-match-only, so every test carries it.
"""
import os
import textwrap

import pytest

from inference import prompts

VSCODE = textwrap.dedent("""\
    You are an AI programming assistant.
    When asked for your name, you must respond with "GitHub Copilot".

    <instructions>
    Use the tools available. Prefer editing over rewriting.
    </instructions>

    <toolUseInstructions>
    Never mention tool names to the user.
    </toolUseInstructions>

    <editFileInstructions>
    Read the file before editing it.
    </editFileInstructions>

    <instructions>
    <skills>
    <skill name="deploy">ship it</skill>
    </skills>
    <agents>
    <agent name="reviewer">reads diffs</agent>
    </agents>
    </instructions>""")


def test_parse_reads_the_shipped_sections(tmp_path):
    text = "<preamble>\nbe terse\n</preamble>\n\n<instructions>\ndo the work\n</instructions>"
    assert prompts.parse(text) == {"preamble": "be terse",
                                   "instructions": "do the work"}


def test_parse_ignores_prose_around_the_sections():
    text = "# notes\nthis line is documentation\n\n<preamble>\nbe terse\n</preamble>\n\nmore prose"
    assert prompts.parse(text) == {"preamble": "be terse"}


def test_split_finds_the_preamble_and_every_block():
    preamble, blocks = prompts.split(VSCODE)
    assert preamble.startswith("You are an AI programming assistant.")
    assert [name for name, _, _ in blocks] == [
        "instructions", "toolUseInstructions", "editFileInstructions",
        "instructions"]


def test_split_of_an_untagged_prompt_is_all_preamble():
    preamble, blocks = prompts.split("just a plain string")
    assert (preamble, blocks) == ("just a plain string", [])


def test_rewrite_replaces_only_the_first_match_of_a_tag():
    """VS Code sends <instructions> twice and the second one carries the
    workspace skills and agents. Replacing both would delete them."""
    out, rewrote = prompts.rewrite(
        VSCODE, {"instructions": "do it my way"}, ["instructions"])

    assert rewrote == ["instructions"]
    assert out.count("do it my way") == 1
    assert "<skills>" in out and "<agents>" in out
    assert "Prefer editing over rewriting" not in out


def test_rewrite_leaves_unnamed_blocks_alone():
    out, _ = prompts.rewrite(
        VSCODE, {"instructions": "x", "toolUseInstructions": "y"},
        ["instructions"])
    assert "Never mention tool names to the user." in out


def test_rewrite_can_replace_the_preamble():
    out, rewrote = prompts.rewrite(
        VSCODE, {"preamble": "You are a terse coding assistant."}, ["preamble"])
    assert rewrote == ["preamble"]
    assert out.startswith("You are a terse coding assistant.")
    assert "GitHub Copilot" not in out
    assert "<toolUseInstructions>" in out


def test_rewrite_shortens_the_prompt():
    sections = {"preamble": "terse.", "instructions": "short."}
    out, _ = prompts.rewrite(VSCODE, sections, ["preamble", "instructions"])
    assert len(out) < len(VSCODE)


def test_rewrite_skips_a_block_the_file_does_not_define():
    """Asked for a block the chosen file has no section for: pass it through
    rather than replacing it with nothing."""
    out, rewrote = prompts.rewrite(VSCODE, {"preamble": "terse."},
                                   ["preamble", "notebookInstructions"])
    assert rewrote == ["preamble"]


def test_rewrite_skips_a_block_the_prompt_does_not_have():
    out, rewrote = prompts.rewrite(VSCODE, {"memoryInstructions": "remember"},
                                   ["memoryInstructions"])
    assert rewrote == []
    assert out == VSCODE


def test_rewrite_with_nothing_selected_is_a_no_op():
    out, rewrote = prompts.rewrite(VSCODE, {"preamble": "terse."}, [])
    assert (out, rewrote) == (VSCODE, [])


def test_available_lists_the_shipped_files():
    files = prompts.available()
    assert files, "inference/prompts/ ships at least one prompt"
    terse = next(f for f in files if f["file"] == "terse-coder.md")
    assert "preamble" in terse["blocks"]


def test_available_skips_files_with_no_sections():
    """prompts/README.md is documentation, not a prompt."""
    assert all(f["file"] != "README.md" for f in prompts.available())


def test_load_refuses_a_path(tmp_path):
    """The dashboard is unauthenticated, so the file name must not be able to
    walk out of the prompts directory."""
    with pytest.raises(ValueError):
        prompts.load("../../etc/passwd")
    with pytest.raises(ValueError):
        prompts.load("/etc/passwd")


def test_load_of_a_missing_file_is_empty():
    assert prompts.load("no-such-prompt.md") == {}


def test_contract_blocks_are_flagged():
    """These tell the client how to shape tool calls and edits; replacing one
    is how you break tool calling, so the UI warns."""
    assert "toolUseInstructions" in prompts.CONTRACT_BLOCKS
    assert "editFileInstructions" in prompts.CONTRACT_BLOCKS
    assert "preamble" not in prompts.CONTRACT_BLOCKS
