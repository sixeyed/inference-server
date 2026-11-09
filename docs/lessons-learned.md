# Lessons learned — engine and model behaviour

Findings from building the inference engine and running real agent sessions
through it, 2026-08-18 to 2026-08-20. Everything here is measured on the M4 Max
(128 GB) against traffic from VS Code Copilot agent mode, not inferred from
documentation.

The short version: **the failures that matter on agentic workloads are
behavioural, not throughput.** Every model here decodes fast enough. What ends
sessions is looping, stalling, re-reading what it already has, or circling — and
none of those show up in a tokens-per-second number.

Three of the four look like something else at first glance: the loop looks like
a broken tool, the re-reading looks like truncation, the stall looks like a hang.
Each section below leads with the measurement that told them apart.

---

## qwen3.8 on MTPLX

### Thinking off, it loops

With `--reasoning off`, qwen3.8 emitted the **byte-identical tool call 165 times
in a row** — 11 minutes, ~65,000 tokens of context added, and it would have run
to the 262K window. An earlier session did the same thing 118 times.

```
grep_search {"query":"BUILD_NUMBER|imageTag|…","includePattern":"scripts/deployment/**"}
```

The conversation grew normally — 2 messages and ~109 tokens per turn — so the
client was behaving. What varied was the tool result, and it didn't matter:

| run | tool result | turns looped |
|---|---|---|
| 1 | `Found 41 matches in 10 files` | 118 |
| 2 | `(empty)` | 165 |

So neither "the tool failed" nor "the tool succeeded" explains it. The model has
165 copies of its own identical call in context and emits a 166th. History
reinforces rather than corrects.

**Four things that could plausibly have broken it, and did not:**

| candidate | why it can't |
|---|---|
| the client's `temperature: 0.1` | stripped at the proxy; model used 0.7/0.8/20 and looped anyway |
| `presence_penalty 1.5` | penalises repetition *within* one response; each turn is a fresh generation |
| MTPLX `repetition_stop` | intra-generation, never fired |
| MTPLX transcript dedup | the counter exists (`transcript_skipped_repeated_assistant_messages`) and reported **0** across 165 identical messages |

VS Code's own loop detection never fired either, across 118 and then 165 turns.

The near-greedy sampling hypothesis is **disproved**, not untested: stripping the
client's parameters was verified end to end (`effective_temperature 0.7,
effective_top_p 0.8`) and the loop continued at genuinely stochastic settings.

### Thinking on, it stalls instead

`--reasoning on --reasoning-effort low` stopped the looping outright — longest
run of identical consecutive tool calls **1**, across 24 turns using three
different tools with varying output sizes. But it introduced runaway turns, in
two distinct shapes:

```
7,687 tokens / 231s   7,442 of them reasoning        a thinking runaway
9,395 tokens / 273s   zero reasoning, zero content   ~250 chars of tool args
```

The second is the stranger one: `reasoning_tokens: 0`, `content: 0 chars`, and
`ttft == ttfc == 277s`. The client saw nothing for four and a half minutes, and
nine thousand tokens never surfaced as anything. MTP acceptance decayed through
the draft depths on that turn — `[0.79, 0.62, 0.49]` with 1,677 correction
tokens, against ~1.0 on healthy turns — which is what a model wandering rather
than converging looks like from the drafter's side.

### With thinking on, it re-reads what it already has

A third behaviour, distinct from the loop and the stall, and easy to mistake for
a tool problem. Over one day's sessions: **80 `read_file` calls across 22
distinct files** — and **25 of them on the same 261-line file**.

The reads are not being truncated. Every one returned exactly the span asked
for, and at one point the model requested lines 1–400 and got the complete file,
all 261 lines and 9,259 characters, with no truncation notice of any kind:

```
1-400   -> 261 lines   (the whole file)
1-130   -> 130
70-265  -> 192
104-260 -> 157
```

Having read the whole thing, it then read the same file **thirteen more times**
in overlapping windows: `1-130`, `1-70`, `70-265`, `104-185`, `185-254`, `1-45`,
`37-95`, `90-120`, `104-260`, `37-109`, `86-110`, `29-40`.

Some of that is legitimate — it interleaves `replace_string_in_file` edits, and
re-reading after an edit to confirm current state is correct. But not thirteen
times, and not in windows that overlap each other rather than targeting what was
just changed.

**What it looks like from the outside is truncation**: each narrow re-read
resembles a partial file, and the next resembles another attempt at the same
thing. It is not. Checking `finish_reason` settles it in one query — no turn in
that session ended `length`, and outputs ran 72–494 tokens against a 6,144 cap.

The context was at 75,000 tokens and climbing, so the file's contents were
present several times over. **Same failure family as the loop**: information
already in context not being used. Worth reaching for the same diagnostic —
compare what was asked for against what came back before assuming a tool is at
fault.

### `max_tokens` is a shared budget, and that is the trap

The obvious fix is a response cap. The first attempt, 2,048, **ended a session**:

```
reasoning tokens   2014      98% of the budget
content chars       159      cut off mid-sentence
tool calls            0
finish_reason      length
```

The model reasoned to a sound conclusion — *ask the user before making a
design-level change* — started writing the answer, and was guillotined 159
characters in with no tool call. The client had nothing to act on and stopped.

**With reasoning on, `max_tokens` covers reasoning and answer together.**
Reasoning spends it first and the reply takes what is left, so a tight cap
converts a slow turn into a turn that produces *nothing* — strictly worse than
the stall it was meant to prevent.

MTPLX offers no reasoning-specific limit: only `--reasoning-effort` levels and a
total `--max-tokens`. There is no setting that bounds thinking without also
bounding the reply.

**6,144** is the working value, and every number in it comes from observed turns:

- above the largest legitimate turn seen (4,700 tokens, 4 tool calls, 4,387
  reasoning tokens)
- above 2,014 reasoning plus a full answer — the case that broke 2,048
- below both runaways (7,687 and 9,395)
- roughly three minutes at this model's ~34 tok/s, against 273s observed

### A truncated reasoning turn wedges the session classifier

Cutting a turn off mid-reasoning with a small `max_tokens` leaves MTPLX
believing a foreground generation is still running. Every subsequent request
whose prompt does not match that session's prefix is then rejected:

```json
{"error": {"message": "background Open WebUI task bypassed while foreground
                       generation is busy", "type": "session_busy"},
 "mtplx_stats": {"cache_miss_reason": "session_busy",
                 "session_restore_mode": "background_bypass"}}
```

It looks like a prompt-content rule and it is not. Requests carrying the *same*
system prompt kept returning 200 indefinitely while `You are a helpful
assistant.` got a 503, which reads as "MTPLX has rejected my new prompt" — the
wrong conclusion entirely. What decides it is only whether the prefix matches
the session MTPLX thinks is mid-generation.

Reproduced with `max_tokens: 16` against a reasoning turn, which the console
line reports as `-> REASONING ONLY`. It does not clear on idle; only reloading
the model does. Two consequences:

- **Never probe a reasoning model with a tiny `max_tokens`.** 64 is enough to
  finish a trivial turn; 16 wedges the server for every other prompt.
- **A 503 `session_busy` is a stuck server, not a rejected request.** Reload the
  model — the engine's `/api/unload` then `/api/load` does it in ~10s — rather
  than editing the prompt that appears to have caused it.

Reasoning is otherwise nearly free: **median 0 tokens, 90th percentile 197**.
Only 2 turns of 25 exceeded 1,000. The cap is not a reasoning budget; it is a
ceiling that only the outliers ever reach.

### `--reasoning off` is real here, unlike on llama.cpp

Worth knowing because the constraint is remembered from the GGUF path: qwen3.8's
llama.cpp chat template accepted only `xhigh`/`medium`/`low` for
`reasoning_effort`, and `none` made the template raise and llama-server 500. So
reasoning genuinely could not be switched off, and `low` was the floor.

MTPLX takes reasoning as a **server flag** rather than through the template, so
`off` is a real setting. Verified rather than assumed — MTPLX's own counters
report `reasoning_tokens: 0`, `request_reasoning_mode: "off"`.

### MTPLX puts its whole diagnostic record on the response

It returns a **llama.cpp-shaped `timings` block** — `predicted_per_second`,
`prompt_ms`, `draft_n`, `draft_n_accepted` — so `_rates()` tags its rows
`source: server` with no special case, and its tok/s is directly comparable with
llama.cpp's.

It also attaches **`mtplx_stats`**, the same ~334-field record it writes to
`~/.mtplx/logs/request-log-<port>.jsonl`: `peak_memory_bytes`, `ttft_s`,
`decode_tok_s`, `mean_accept_probability_by_depth`, `session_cache_hit`. Handed
over directly, so nothing needs to tail that file.

One catch: `sse.reassemble()` originally dropped it, because it only carried
standard fields through. Streamed turns — which is every real client — silently
lost all of it. It now carries any non-standard block on the final chunk.

### `mtplx serve` is a wrapper

It re-execs into `python -m mtplx.server.openai` with a translated flag set
(`--mtp` → `--generation-mode mtp`, `--reasoning off` →
`--reasoning-mode off --no-enable-thinking`, `--profile turbo` → the verify
strategy and draft-head settings). **The process you spawn is not the process
that listens.**

Consequences, all of them load-bearing:

- `Popen.poll()` on the wrapper is not a liveness signal for the server
- `start_new_session=True` plus `killpg` is mandatory, not defensive
- memory must be sampled from `find_server_pid(port)`, never `popen.pid`
- `mtplx stop` **does not work** against it — that command only manages the
  daemon `mtplx start` registers, and exits 1 with *"Port 8083 is in use, but
  not by an MTPLX server. Not touching it."* Plain SIGTERM is what stops it, and
  MTPLX handles SIGTERM cleanly.

---

## kat-coder2.5

**Fast, but circles — and the circling is measurable.** 55.4 tok/s pooled and
3.7s average TTFT, comfortably the quickest of the three. On a real task it
produced a lot of tool calls and little code, and the numbers say the same
thing:

| | qwen3.8 | kat-coder2.5 |
|---|---|---|
| turns measured | 764 | 29 |
| mean output per turn | 218 tokens | **1,832 tokens** |
| tool calls per turn | 1.06 | **2.72** |
| output tokens per tool call | 206 | **673** |
| turns over 5,000 tokens | 3 | **4** |
| largest turn | 9,395 | **11,619** |

Eight times the output per turn and three times the output per tool call. It is
generating a great deal of text per unit of action — which is what "going around
the problem" looks like in a table. Note the sample: 29 kat turns against 764,
so treat the ratios as indicative rather than settled.

Its runaways are also **unbounded**: it runs on llama-server with
`predict=32768` and no equivalent of the 6,144 cap applied to qwen3.8's thinking
profile. Four of its 29 turns exceeded 5,000 tokens, the longest running 196
seconds.

Different failure from qwen3.8's, and not one throughput exposes.

**It says it is Qwen.** Asked "who are you?", it answers *"I am Qwen, a large
language model independently developed by Alibaba Group's Tongyi Lab."* It is
built on Qwen3.6-35B-A3B, and Kwaipilot's SFT and RL targeted agentic coding
behaviour, not self-identity. **Asking a model who it is tells you what it was
trained from, not what you are running.** Four independent signals confirmed the
routing was correct: the loaded GGUF path, 39.5 GB resident (vs ~24 GB for
qwen3.8), 73 tok/s at 51 ms TTFT, and `mtp_accept: null` — no speculative stats,
because MTP is an MTPLX feature.

---

## muse-glimmer

**Garbage output was a corrupt download, not a version problem.** The model
emitted runs of `S` characters, llama.cpp's PEG parser rejected them, and the
server exited:

```
W common_chat_peg_parse: unparsed peg-native output: SSSSSSSSSSSSSSSS…
E 500 "The model produced output that does not match the expected peg-native format"
I srv operator(): cleaning up before exit...
```

It failed identically with `--no-spec` and on the raw `/completion` endpoint, so
neither speculative decoding nor the chat template was at fault. Upgrading
llama.cpp 10450 → 10470 did not fix it either.

The cause: a download interrupted at 25.79 of 30.1 GB, then resumed. The file
was **exactly the right size and wrong inside**.

**HuggingFace blob filenames are the content sha256**, so this is checkable in
one command:

```bash
shasum -a 256 ~/.cache/huggingface/hub/models--…/blobs/<name>
# expected  e63bf23b7710ecdea2579e4b1de58980c4a2b446e8ecf48b782cfcefd2e31770
# actual    60dce052a44954ad4826f98f6aa7aaec17cb3136b5f9b8fe8a57f31651014d6f
```

Verifying the whole cache took under two minutes and found exactly one bad blob
out of eight. **Check the hash before diagnosing the model.**

**Context was configured beyond the training length.** The model file asked for
262,144 where `n_ctx_train` is 131,072; llama.cpp capped it silently and only
said so in a log line. The banner and dashboard were reporting a number the
model never had. Now set to the training length.

**llama.cpp downloads into the HuggingFace cache**, not `~/Library/Caches/llama.cpp`
as an older comment claimed. That matters because `find_in_hf_cache()` reads the
HF cache: once weights land there, `argv()` switches from `-hf <repo>` to
`-m <path>` and the launch needs no network at all — which is what makes the
egress sandbox workable.

---

## VS Code Copilot as a client

**It overrides your sampling on every request:** `temperature: 0.1, top_p: 1`.
Whatever a model's file tuned is discarded unless the proxy strips those keys.
Near-greedy decoding is a plausible contributor to agentic loops in general,
though it was **not** the cause of qwen3.8's (see above).

**It offers 51–80 tools**, and `copilot_getNotebookSummary` appears in every
request captured here — a reliable fingerprint across every tool count observed.
Replaying detection over 745 captured turns labelled all of them correctly.

**It does not stream-assemble `reasoning_content`.** MTPLX emits reasoning one
committed token per SSE event (`--stream-interval 1`), so deltas average **4.2
characters** and routinely break mid-word:

```
['T', 'he user ', 'want', 's me ', 'to thin', 'k bri', 'efl', 'y and ']
```

VS Code renders each event as a separate bullet, so the model's reasoning
appears chopped into fragments. Its *content* deltas are equally small and
render fine, so this is specific to the non-standard `reasoning_content` field.

**This is an engine artifact, not a proxy one** — proved by capturing the same
request from the model port and through the engine and comparing: identical 22
deltas, identical 3.9-char mean, identical 7 mid-word splits, identical bytes.
The lever is MTPLX's `--stream-interval` (currently 1), which sets the committed-
token batch size per SSE chunk.

**Its loop detection never fired** across 118 and 165 identical turns.

**Its system prompt is 15,126 characters**, in seven tagged blocks over a 565-
character untagged preamble. A local model pays for that in context and in
attention spent on instructions written for a frontier model, which is why the
engine can substitute named blocks of it (see the README).

The trap in doing that: **`<instructions>` appears twice**, at offset 565 and
again at 10,284. The second one is not a duplicate — it wraps the workspace's
`<skills>` and `<agents>`, so it is the longest block in the prompt (4,383
chars) and it is the one carrying anything you configured yourself. A naive
replace-all deletes your skills and reports success. Substitution is therefore
first-match-only, and the tests carry a fixture with the duplicate in it so a
later change cannot quietly regress that.

The two blocks worth leaving alone are `<toolUseInstructions>` and
`<editFileInstructions>`: they describe the shape VS Code parses replies
against, so replacing one breaks tool calling rather than shortening a prompt.
Replacing the preamble and the first `<instructions>` is the safe cut, and it is
worth about 2,000 characters.

---

## Measuring this stuff

**Pool rates, never average them.** A model's tok/s is total tokens over total
decode time. Averaging per-request rates over-weights short turns, and an agent
session is mostly short turns punctuated by a few long ones. The same applies to
MTP acceptance: pooling gave **0.521** where averaging per-turn ratios gave
**0.761**, because a 4-token reply with one verify round counted as heavily as a
197-token generation.

**Exclude turns that did not finish.** A turn cut by a model swap has a real
token count and a meaningless decode rate — it stopped mid-generation. Counting
it drags every average down by an amount that depends on how often you switched
models, which is the least interesting variable in the data.

**Read `rate_source` alongside any tok/s figure.** llama.cpp and MTPLX both
publish their own decode timers; anything without one falls back to
wall-minus-TTFT arithmetic that includes queueing. Those are different
measurements and should never be compared silently.

**The session is the unit, not the turn and not all-time.** A pooled average
across every turn ever served answers no question worth asking. Sessions are
recoverable from the traffic without any client cooperation: every turn resends
the whole conversation, so the opening message is identical within a session and
differs between them, and a drop in message count is a new session starting.
Verified against 230 captured turns — the opening hash changed exactly where the
session did.

**`ttfc − ttft` is the deferral cost** — how long a thinking model reasons before
the client has anything to show. It is the number that makes a stall visible;
a 277-second turn and a 5-second turn have similar decode rates.

**A cap on one model is not a cap on the fleet.** The 6,144 ceiling applies to
qwen3.8's thinking profile only. kat-coder2.5 and muse-glimmer run on
llama-server with `predict=32768` and nothing bounds a runaway there — kat's
longest turn was 11,619 tokens over 196 seconds.

---

## Operating the thing

**Never kill a download.** Interrupting one at 25.79 GB produced a right-sized
corrupt file that cost an unnecessary llama.cpp upgrade and a wrong diagnosis
before the checksum settled it.

**Name-based process sweeps reach further than you think.** `pgrep -f
llama-server` matches a weights download running on another port — that is how
the download above died. And naming the package `inference` made `pgrep -f
inference` match macOS's own `siriinferenced`. Scope a stop by the port a
process was launched with, and locate the engine by the socket it owns rather
than by its name.

**A long-running Python process holds the code it started with.** After a
session of edits the engine is the last thing to reflect them; comparing source
mtimes against the process start time is how to tell. A stale engine reported a
model that was not loaded for several minutes before anyone noticed.

**Failed loads are worth recording.** A model that fails to start writes to
`logs/<alias>-<timestamp>.log` and the last 30 lines belong in the error the
client sees — Cline and Copilot both render `error.message`, and it is the only
place a user will ever see `unknown flag --spec-type`.

**Refuse a launch that cannot work before stopping the one that does.** With the
egress sandbox on and weights absent, the download is blocked and the launch is a
*known* failure. Discovering that after the swap costs a working session to learn
something knowable beforehand.
