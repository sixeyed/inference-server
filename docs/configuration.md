# Configuration

Three kinds of setting, kept apart because they change on different timescales.

**`inference/models/*.py` — how a model is deployed.** Quant per machine,
context, sampling, thinking switches, speculative decoding. One file tells you
the whole deployment for one model.

**`config.json` — where this machine is.** Paths, addresses, per-chip tuning.
Gitignored, because it describes your machine. Copy `config.example.json` and
edit. Omit anything you don't need and it falls back to a default that works on
a fresh clone.

This one serves the network — the defaults keep everything on `127.0.0.1`:

```json
{
  "weights_root": "/Volumes/your-ssd/weights",
  "bind": "0.0.0.0",
  "api_key": "something-not-none",
  "llama_load_mode": "mmap+mlock",
  "machines": {
    "m4": { "chip": "M4 Max", "host": "192.168.1.10", "threads": 12,
            "batch": 4096, "ubatch": 2048, "cache_ram_mb": 32768 }
  }
}
```

| Key | Default | |
|---|---|---|
| `bind` | `127.0.0.1` | where the engine listens; `0.0.0.0` serves your network |
| `model_bind` | `127.0.0.1` | where model servers listen; only open it up when the engine runs on another machine |
| `auto_pull` | `true` | fetch missing weights before launch; `false` refuses instead |
| `weights_root` | `~` | where the HuggingFace and MTPLX caches go |
| `api_key` | `none` | the one key for clients and model servers |
| `machines.<chip>.host` | `127.0.0.1` | the address the dashboard tells clients to use |
| `client_endpoints` | this machine | the addresses the dashboard's Clients tab offers |
| `model_host` | none | drive a Mac's model servers over SSH — see [Running in a container](engine.md#running-in-a-container) |

Everything listens on loopback until you say otherwise. If you open `bind` up,
set a real `api_key` too: the default `none` is no auth at all.

The chip is still detected; `machines` only supplies what follows from it, so a
64GB M4 Max is a config change rather than a code change. Precedence is
environment variable, then `config.json`, then the built-in default —
`INFERENCE_WEIGHTS_ROOT`, `INFERENCE_BIND`, `INFERENCE_MODEL_BIND`,
`INFERENCE_AUTO_PULL`, `INFERENCE_CONFIG`, `HF_HOME`, `HF_TOKEN` and
`MTPLX_API_KEY` all win over the file, because setting one is a deliberate act at the moment of
running something. One more is only for running off a Mac: `INFERENCE_MACHINE`
names the profile to stand in when there is no chip to detect — the image's
test stage sets it.

**`settings.json` — how the proxy behaves right now.** Written by the dashboard's
Config tab, reloaded on restart. See
[Stripping client parameters](#stripping-client-parameters) and the sections
after it.

## Weights on another volume

`weights_root` moves both caches at once: llama-server and mlx_lm.server resolve
through the HuggingFace layout (`HF_HOME`), and MTPLX keeps its own flat cache
beside it. The child process inherits the same environment, so the launcher and
the engine cannot disagree about where the weights are.

The MTPLX **session bank** stays on the internal disk deliberately. Unlike the
weights it is read *during* serving — restored on prefill, which is the current
bottleneck — and it is budgeted to grow to 48GB. It is the one thing here that
actually wants the fast disk.

`llama_load_mode` defaults to `mmap+mlock`: llama.cpp maps the GGUF then pins it
in RAM, so weights on a slower external disk are read once at load and never
paged back in mid-request. That costs locked RAM — drop to `auto` on a machine
smaller than 128GB.

If `weights_root` is on a volume that is not mounted the engine says so and
stops, rather than letting
`find_in_hf_cache()` return nothing and llama-server quietly start
re-downloading tens of gigabytes to a path that isn't there.

## Stripping client parameters

Clients send their own sampling settings, which override whatever the model's
file tuned it to — VS Code Copilot sends `temperature: 0.1, top_p: 1` on every
request. Near-greedy decoding is how an agent session gets stuck re-emitting an
identical tool call: each turn's context differs only by an identical
assistant/tool pair, so a deterministic decoder re-derives the same answer.

The dashboard's **Config** tab lists the parameters that can be stripped. Strip
one and the key is removed on the way through, so the model's own default
applies. It takes effect on the next request with no restart, and it is
remembered across restarts — a setting you had to remember to pass on the
command line is one you will not have set when it matters.

Strip only, never override: sampling stays defined in one place, the model's own
file. `model`, `messages`, `tools`, `tool_choice`, `stream` and `stream_options`
are protected, because removing those changes what was asked rather than how it
was sampled. The trace records `stripped_params` and keeps the client's original
values in `request_original`.

## Rewriting the system prompt

VS Code sends about 15,000 characters of Copilot framing on every turn. A local
model pays for that twice — in context, and in attention spent on instructions
written for a frontier model. The prompt from `traces/turn-1-request.json`
breaks down like this:

| offset | block | chars |
|---|---|---|
| 0 | *untagged preamble* | 565 |
| 565 | `<instructions>` | 2,257 |
| 2,823 | `<toolUseInstructions>` | 1,733 |
| 4,557 | `<editFileInstructions>` | 2,306 |
| 6,864 | `<notebookInstructions>` | 912 |
| 7,777 | `<outputFormatting>` | 575 |
| 8,353 | `<memoryInstructions>` | 1,929 |
| 10,284 | `<instructions>` | 4,383 |

The dashboard's **Config** tab substitutes named blocks of that, per client,
from the files in `inference/prompts/`. Pick a file and tick the blocks it
should replace; anything you don't tick is forwarded exactly as the client sent
it. Replacing the preamble and the first `<instructions>` with the shipped
`terse-coder.md` takes the prompt above from 15,126 to 13,120 characters.

Two things it deliberately does not do:

- **Only the first match of a tag is replaced.** That last row is not a
  duplicate — VS Code sends `<instructions>` twice, and the second one wraps the
  workspace's `<skills>` and `<agents>`. Replacing both would silently delete
  the skills you configured.
- **It never replaces a block the file has no section for.** Asking for a block
  the prompt doesn't contain is a no-op too, so a file written for VS Code does
  nothing surprising when Cline connects.

`<toolUseInstructions>` and `<editFileInstructions>` are flagged **contract** in
the UI: they describe the shape the client parses replies against, so replacing
one is how you break tool calling rather than how you shorten a prompt. They can
be replaced, but that is a thing to do on purpose.

Prompt files are read per request, so editing one takes effect on the next turn
without restarting the engine — which is the only way iterating on a prompt is
bearable. Each file is one system prompt with a `<name>` section per block it
replaces; see `inference/prompts/README.md`. The trace records `rewrote_prompt`
(the file, the blocks, and the character counts either side) and keeps the
client's original message in `request_original`, so a trace never claims the
model saw something it didn't.

## Filtering the tool list

The prompt is the smaller half of what a client sends. VS Code also offers its
whole tool surface on every turn — 80 definitions in the most recent captured
session, about 23,000 tokens of JSON schema, resent in full on every request of
every session.

Across the 15 trace files in `traces/` it offered **103 distinct tools and the
model called 18**:

| tool | calls | | tool | calls |
|---|---:|---|---|---:|
| `grep_search` | 436 | | `create_file` | 14 |
| `read_file` | 411 | | `fetch_webpage` | 14 |
| `run_in_terminal` | 304 | | `manage_todo_list` | 11 |
| `replace_string_in_file` | 95 | | `github_text_search` | 3 |
| `list_dir` | 35 | | `get_terminal_output` | 2 |
| `task_complete` | 29 | | `github_repo` | 2 |
| `file_search` | 20 | | `insert_edit_into_file` | 2 |
| `get_errors` | 15 | | `vscode_askQuestions` | 1 |
| `memory` | 15 | | `vscode_searchExtensions_internal` | 1 |

The other 85 are ~76% of the payload, and they cost twice: context the
conversation cannot use, and 85 wrong answers for the model to choose between
every time it decides what to call next. Notable dead weight — around 30
dotnet-SDK-management tools, the Pylance MCP set, every notebook and browser
tool, `runSubagent`, `vscode_renameSymbol`, `session_store_sql`.

A **tool set** names the tools a client may offer. The dashboard's **Config**
tab has one per client, beside the prompt picker: pick a set and the rest are
dropped on the way through. `inference/toolsets/vscode-core.txt` is those 18,
less the two called exactly once, plus three companions the traces would not
show until something goes wrong (`get_terminal_output` for a backgrounded
terminal, `insert_edit_into_file` for when `replace_string_in_file` fails,
`resolve_memory_file_uri` for `memory`). On a real captured turn that is 66
tools down to 16 — ~19,600 schema tokens down to ~6,900, about 30% off the whole
request.

An allowlist, not a patch: a name the client didn't send is reported in the
trace rather than added, because the engine has no definition to add. And a set
that matches nothing is treated as a typo — the client's tools go through
untouched, because zero tools doesn't make a coding agent fail, it makes it
quietly stop being an agent.

This is the one place the engine changes *what was asked* rather than how it is
sampled, which is why `tools` stays protected in the strip list and this is a
separate, opt-in, per-client mechanism. Sets are read per request, so editing
one takes effect on the next turn. The trace records `filtered_tools` (the file,
what was kept, what was dropped, and any name the client no longer offers) and
keeps the full array in `request_original`.

## Breaking a stuck tool-call loop

On 27 August a session asked about ARM builds got stuck. The model built a
malformed Azure DevOps URL — `[char]38` (`&`) where the first query parameter
needs `?` — and re-issued the byte-identical command **37 times**. By the last
turn, 74 of the 102 messages were that same wrong command and that same 404.

Sampling cannot fix this, and it is worth being precise about why:

- Temperature was already stripped, so the model's own 0.7 applied — not
  greedy. Replaying the stuck turn still gave byte-identical output **6/6**.
  The distribution in that context is peaked, not noisy.
- Temperature is a per-token knob; the repetition is *across turns*. Each turn
  is a fresh sample from a context that genuinely leads to the same place.
- `presence_penalty` has the same blind spot — it only sees within one
  generation.

It is not a knowledge problem either. The same model asked cold produces the
correct URL every time. It is context poisoning: the transcript contains 37
examples of the wrong answer, immediately before the question.

The engine is the only component positioned to fix it. Chat completions is
stateless, so the whole history arrives every turn, and the client — VS Code
here — did not stop it after thirty-seven tries.

**What the fix had to look like was measured, not guessed.** Replaying the
stuck turn against the real model:

| what the model is left with | escapes the loop |
|---|---|
| one copy of the call, plus an explanatory note | 0 / 3 |
| one copy of the call, plus a *directive* note | 2 / 3 |
| no copy, note only | 3 / 3 |
| **no copy, with the shared result kept in the note** | **4 / 4** |

The anchor is the command text, not the error. An example of the wrong command
in front of the model is an example it copies, however firmly the note tells it
not to. So the run is replaced by a single message reporting what came back and
never what was sent — on the real turn, 102 messages down to 29.

Set a trigger count per client on the **Config** tab; `0` is off. Two things
keep a genuine poll safe: the fingerprint covers the tool *results*, so a poll
whose output moves on (`queued` → `running`, an elapsed time, a percentage) is
never a match at any threshold; and the count is there for one whose output
doesn't. Runs are collapsed wherever they appear, not only at the tail —
otherwise the moment the model escapes, its new call makes the run non-trailing
and the whole loop comes back on the next turn.

The trace records `collapsed_loops` and keeps the full history in
`request_original`, and every intervention gets its own row on the Logs page —
undeduped, unlike the other events, because each one is the engine changing a
live conversation.

## The Logs page

The engine changes things on the way past — it strips parameters, rewrites
prompt blocks, filters tools, repairs tool arguments. All of that used to be a
console line printed once per engine run, guarded by a boolean set on the first
occurrence and never cleared. That reported the *first* state and hid every
later one, which is backwards: the case worth seeing is the one that appears
halfway through a session.

The **Logs** tab records an event when something *changes*, keyed on content
rather than a flag. A long session with steady settings writes nothing; a
client update that changes what it sends writes a line. Rows live in
`usage.db`, so they survive the restart you do straight after an update, and
failed model loads are merged in on read from `loads` rather than copied, so a
failure can't go stale in a second copy. There is no retention job — deduping
makes this tens of rows a week, and a cleaner would only delete the history you
opened the page to read.

**Drift** is the line to act on, and it is narrower than "the set names a tool
the client didn't send". A set entry the client has *never* offered has simply
never applied — reporting it would light the indicator permanently, which is
the same as not having one. Drift is a tool the client **did** offer and has
stopped offering: the model has silently lost it, and the guard doesn't fire
because every other name still matches. That is what a rename in a client
release looks like from here — and VS Code really does rename tools, having
moved the whole Pylance family from `mcp_pylance_mcp_s_*` to
`mcp_provides_tool_*` between 20 and 26 August in the captures here.

The Config tab shows the same thing against the client it affects, next to the
tool set that asks for it. The engine can't detect drift for a tool it has
never seen, which is honest: on a fresh `usage.db` there is nothing to have
drifted from.
