"""What each model actually did, in SQLite.

The JSONL traces hold the whole conversation and are the thing you read when you
want to know *what happened*. This holds one row per turn and is the thing you
query when you want to know *which model is faster*. `trace_file` and
`trace_seq` join the two, so any row here can be taken back to the exact
exchange that produced it.

Nothing new is measured. The proxy already times the stream and the servers
already report their own token rates; this keeps them.

Two rules that decide whether the numbers mean anything:

Rates are **pooled**, never averaged. A model's tok/s is total tokens over total
decode time - averaging per-request rates over-weights short requests, which is
the same trap tests/stats.py::merge() documents.

**Preempted turns are excluded from every aggregate.** A turn cut by a model
swap has a real token count and a decode rate that means nothing, because it
stopped mid-generation. Counting it would drag every average down by an amount
that depends on how often you switched models.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import threading
import time
from datetime import datetime, timezone

SCHEMA = """
CREATE TABLE IF NOT EXISTS requests (
  id             INTEGER PRIMARY KEY,
  ts             TEXT NOT NULL,
  alias          TEXT NOT NULL,
  requested      TEXT,
  client_ip      TEXT,
  session_id     TEXT,
  client         TEXT,
  engine         TEXT NOT NULL,
  path           TEXT NOT NULL,
  stream         INTEGER NOT NULL,
  ok             INTEGER NOT NULL,
  status         INTEGER,
  preempted      INTEGER NOT NULL DEFAULT 0,
  ttfb_ms        REAL, ttft_ms REAL, ttfc_ms REAL, total_ms REAL,
  deferral_ms    REAL,
  prompt_tokens  INTEGER, cached_tokens INTEGER, output_tokens INTEGER,
  context_tokens INTEGER,
  prefill_ms     REAL, decode_ms REAL,
  prefill_tps    REAL, decode_tps REAL, rate_source TEXT,
  ttft_ms_server REAL,
  upstream_id    TEXT,
  peak_rss_mb    REAL,
  mtp_depth      INTEGER, mtp_accept REAL, mtp_bonus_tokens INTEGER,
  mtp_drafted    INTEGER, mtp_accepted INTEGER,
  cache_hit      INTEGER, cache_source TEXT,
  n_messages     INTEGER, n_tools INTEGER, n_tool_calls INTEGER,
  tool_names     TEXT,          -- what was called, in order
  tool_args      TEXT,          -- hash per call, to spot a repeat
  finish_reason  TEXT,
  load_ms        REAL,
  trace_file     TEXT, trace_seq INTEGER
);

CREATE TABLE IF NOT EXISTS loads (
  id          INTEGER PRIMARY KEY,
  ts          TEXT, alias TEXT, engine TEXT,
  load_ms     REAL, ok INTEGER, error TEXT, log_path TEXT,
  peak_rss_mb REAL, unloaded_at TEXT
);

CREATE TABLE IF NOT EXISTS memory (
  ts TEXT, alias TEXT, rss_mb REAL
);

CREATE TABLE IF NOT EXISTS events (
  id     INTEGER PRIMARY KEY,
  ts     TEXT NOT NULL,
  kind   TEXT NOT NULL,
  client TEXT,
  text   TEXT,
  detail TEXT
);
"""

# Indexes are applied *after* the column migration, not inside SCHEMA. An index
# naming a column an older database has not got yet makes executescript() fail,
# which aborts before the migration that would have added it - so a new indexed
# column bricks every existing usage.db.
INDEXES = """
CREATE INDEX IF NOT EXISTS requests_alias_ts ON requests (alias, ts);
CREATE INDEX IF NOT EXISTS requests_session  ON requests (session_id);
CREATE INDEX IF NOT EXISTS memory_ts         ON memory (ts);
CREATE INDEX IF NOT EXISTS events_ts         ON events (ts);
"""

MEMORY_RETENTION_DAYS = 7


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ============================================================
# Turning a response into numbers
# ============================================================
def rates(response: dict, ttft_ms: float | None, wall_ms: float) -> dict:
    """Prefer the server's own timings; fall back to client-side arithmetic.

    Which was used is returned as `source`, because the two are not
    interchangeable: the server's figures exclude HTTP and queueing, the
    client's don't. Both llama.cpp and MTPLX publish `timings` (MTPLX's is
    llama.cpp-shaped, down to `predicted_per_second`), so in practice only
    mlx_lm.server takes the fallback.
    """
    usage = response.get("usage") or {}
    timings = response.get("timings") or {}

    prompt_tokens = timings.get("prompt_n", usage.get("prompt_tokens"))
    output_tokens = timings.get("predicted_n", usage.get("completion_tokens"))

    if timings.get("predicted_per_second"):
        return {
            "source": "server",
            "prompt_tokens": prompt_tokens,
            "output_tokens": output_tokens,
            "cached_tokens": timings.get("cache_n"),
            "prefill_tps": round(timings.get("prompt_per_second") or 0, 1) or None,
            "decode_tps": round(timings["predicted_per_second"], 1),
            "prefill_ms": round(timings.get("prompt_ms") or 0, 1) or None,
            "decode_ms": round(timings.get("predicted_ms") or 0, 1) or None,
        }

    # No server timings. Decode time is wall minus the time spent before the
    # first token, which is prefill plus queueing.
    decode_ms = wall_ms - (ttft_ms or 0)
    decode_tps = (output_tokens / (decode_ms / 1000)
                  if output_tokens and decode_ms > 0 else None)
    prefill_tps = (prompt_tokens / (ttft_ms / 1000)
                   if prompt_tokens and ttft_ms else None)
    return {
        "source": "client",
        "prompt_tokens": prompt_tokens,
        "output_tokens": output_tokens,
        "cached_tokens": None,
        "prefill_tps": round(prefill_tps, 1) if prefill_tps else None,
        "decode_tps": round(decode_tps, 1) if decode_tps else None,
        "prefill_ms": round(ttft_ms, 1) if ttft_ms else None,
        "decode_ms": round(decode_ms, 1) if decode_ms > 0 else None,
    }


def mtplx_extras(response: dict) -> dict:
    """The bits only MTPLX reports, read defensively.

    MTPLX puts its whole per-request diagnostic record on the response as
    `mtplx_stats` - the same 300-odd fields it writes to
    ~/.mtplx/logs/request-log-<port>.jsonl, but handed to us directly, so
    nothing here depends on finding or tailing that file.

    The names are still MTPLX's own and an upgrade could rename them, so every
    read is a .get() and the caller keeps the standard `timings` values when
    these are missing.
    """
    stats = response.get("mtplx_stats") or {}
    if not stats:
        return {}

    drafted = stats.get("drafted_by_depth") or []
    accepted = stats.get("accepted_by_depth") or []
    # Keep the raw counts, not just the ratio. Acceptance has to be pooled
    # across requests the same way token rates are - a two-token reply is one
    # verify round, and averaging its ratio against an 800-token turn's says
    # more about how short your test prompts were than about the model.
    n_drafted, n_accepted = sum(drafted) or None, sum(accepted) if drafted else None
    accept = round(n_accepted / n_drafted, 3) if n_drafted else None

    peak = stats.get("peak_memory_bytes")
    ttft_s = stats.get("ttft_s")
    return {
        "ttft_ms_server": round(ttft_s * 1000, 1) if ttft_s else None,
        "peak_rss_mb": round(peak / 1024**2, 1) if peak else None,
        "mtp_depth": stats.get("mtp_depth"),
        "mtp_accept": accept,
        "mtp_drafted": n_drafted,
        "mtp_accepted": n_accepted,
        "mtp_bonus_tokens": stats.get("bonus_tokens"),
        "cache_hit": int(bool(stats.get("session_cache_hit"))),
        "cache_source": stats.get("cache_source"),
        "cached_tokens": stats.get("cached_tokens"),
    }


# ============================================================
# The database
# ============================================================
class UsageDB:
    def __init__(self, path: str, machine=None, sample_interval: float = 5.0):
        self.path = path
        self.machine = machine
        self.sample_interval = sample_interval
        self._lock = threading.Lock()
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        # WAL so the dashboard polling for stats never blocks a handler thread
        # in the middle of a turn.
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.executescript(SCHEMA)
        self._migrate()
        self._db.executescript(INDEXES)
        self._db.commit()
        self._prune_memory()

        self._sampler: threading.Thread | None = None
        self._sampling = threading.Event()
        self._load_id: int | None = None
        self._peak_rss = 0.0
        self._supervisor = None

    def _migrate(self) -> None:
        """Add columns that SCHEMA has gained since this database was created.

        `CREATE TABLE IF NOT EXISTS` does nothing to a table that already
        exists, so without this every new column fails silently on an existing
        usage.db - the insert errors, the row is lost, and because stats
        failures are swallowed on purpose you only find out by reading the
        console. Deriving the expected columns from SCHEMA keeps adding one a
        single edit.

        Split on commas rather than on lines: a declaration like
        `mtp_depth INTEGER, mtp_accept REAL` is two columns on one line, and
        parsing it line-wise produces an ALTER TABLE that SQLite rejects.
        """
        for table, body in re.findall(
                r"CREATE TABLE IF NOT EXISTS (\w+) \((.*?)\n\);", SCHEMA, re.S):
            have = {row[1] for row in
                    self._db.execute(f"PRAGMA table_info({table})")}
            # Strip comments *before* splitting on commas. A comment
            # containing a comma - "-- what was called, in order" - otherwise
            # splits into fragments and the parser tries to add a column called
            # "in".
            clean = "\n".join(line.split("--")[0].rstrip()
                               for line in body.splitlines())
            for fragment in clean.split(","):
                fragment = " ".join(fragment.split()).strip()
                if not fragment:
                    continue
                name, _, decl = fragment.partition(" ")
                if name in have or not decl or not name.isidentifier():
                    continue
                # Only ever additive, and never with a NOT NULL that has no
                # default - SQLite refuses that on a populated table, and a
                # migration that can fail is worse than a missing column.
                decl = decl.replace("NOT NULL", "").strip()
                self._db.execute(f"ALTER TABLE {table} ADD COLUMN {name} {decl}")
                print(f"  usage: added {table}.{name}", flush=True)

    # --------------------------------------------------------
    def _write(self, sql: str, params: tuple):
        """Never let a stats failure cost a turn. The point of the box is
        serving models; losing a row is not worth losing a response."""
        try:
            with self._lock:
                cursor = self._db.execute(sql, params)
                self._db.commit()
                return cursor.lastrowid
        except sqlite3.Error as exc:
            print(f"  usage: {exc}", flush=True)
            return None

    def _query(self, sql: str, params: tuple = ()) -> list[dict]:
        try:
            with self._lock:
                return [dict(r) for r in self._db.execute(sql, params).fetchall()]
        except sqlite3.Error as exc:
            print(f"  usage: {exc}", flush=True)
            return []

    def _prune_memory(self) -> None:
        cutoff = datetime.now(timezone.utc).timestamp() - MEMORY_RETENTION_DAYS * 86400
        stamp = datetime.fromtimestamp(cutoff, timezone.utc).isoformat(
            timespec="seconds")
        self._write("DELETE FROM memory WHERE ts < ?", (stamp,))

    # --------------------------------------------------------
    # Writing
    # --------------------------------------------------------
    def record_request(self, *, seq, alias, requested, current, path, streaming,
                       timing, request, response, preempted, trace_file,
                       client_ip=None, session_id=None, client=None):
        request = request or {}
        response = response or {}
        messages = request.get("messages") or []
        tools = request.get("tools") or []
        choice = (response.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        calls = message.get("tool_calls") or []
        status = response.get("_http_error")
        ok = not (status or response.get("_interrupted"))

        figures = rates(response, timing.get("ttft_ms"), timing.get("total_ms") or 0)
        extras = mtplx_extras(response)

        usage = response.get("usage") or {}
        details = usage.get("prompt_tokens_details") or {}
        cached = (extras.get("cached_tokens")
                  if extras.get("cached_tokens") is not None
                  else figures.get("cached_tokens"))
        if cached is None:
            cached = details.get("cached_tokens")

        context = usage.get("prompt_tokens")
        if context is None and figures.get("prompt_tokens") is not None:
            context = (figures["prompt_tokens"] or 0) + (cached or 0)

        ttft, ttfc = timing.get("ttft_ms"), timing.get("ttfc_ms")
        deferral = round(ttfc - ttft, 1) if (ttft and ttfc) else None

        self._write(
            """INSERT INTO requests (
                 ts, alias, requested, engine, path, stream, ok, status, preempted,
                 ttfb_ms, ttft_ms, ttfc_ms, total_ms, deferral_ms,
                 prompt_tokens, cached_tokens, output_tokens, context_tokens,
                 prefill_ms, decode_ms, prefill_tps, decode_tps, rate_source,
                 ttft_ms_server, upstream_id, peak_rss_mb,
                 mtp_depth, mtp_accept, mtp_bonus_tokens,
                 mtp_drafted, mtp_accepted, cache_hit, cache_source,
                 n_messages, n_tools, n_tool_calls, tool_names, tool_args,
                 finish_reason, load_ms,
                 trace_file, trace_seq, client_ip, session_id, client)
               VALUES (?,?,?,?,?,?,?,?,?, ?,?,?,?,?, ?,?,?,?, ?,?,?,?,?,
                       ?,?,?, ?,?,?,?,?,?,?, ?,?,?,?,?,?,?, ?,?,?,?,?)""",
            (_now(), alias, requested, current.engine, path, int(streaming),
             int(ok), status, int(preempted),
             timing.get("ttfb_ms"), ttft, ttfc, timing.get("total_ms"), deferral,
             figures.get("prompt_tokens"), cached, figures.get("output_tokens"),
             context,
             figures.get("prefill_ms"), figures.get("decode_ms"),
             figures.get("prefill_tps"), figures.get("decode_tps"),
             figures.get("source"),
             extras.get("ttft_ms_server"), response.get("id"),
             extras.get("peak_rss_mb"),
             extras.get("mtp_depth"), extras.get("mtp_accept"),
             extras.get("mtp_bonus_tokens"),
             extras.get("mtp_drafted"), extras.get("mtp_accepted"),
             extras.get("cache_hit"), extras.get("cache_source"),
             len(messages), len(tools), len(calls),
             ",".join(c.get("function", {}).get("name") or "?" for c in calls) or None,
             # A short hash per call rather than the arguments themselves: the
             # detail page only needs to know whether two turns asked for the
             # same thing, and arguments can be kilobytes of file content.
             ",".join(hashlib.sha1(
                 (c.get("function", {}).get("arguments") or "").encode()
             ).hexdigest()[:8] for c in calls) or None,
             choice.get("finish_reason"), current.load_ms or None,
             os.path.basename(trace_file), seq, client_ip, session_id,
             client))

    def record_load(self, current) -> None:
        self._load_id = self._write(
            """INSERT INTO loads (ts, alias, engine, load_ms, ok, log_path)
               VALUES (?,?,?,?,1,?)""",
            (_now(), current.alias, current.engine, current.load_ms,
             current.log_path))
        self._peak_rss = 0.0
        self._start_sampling(current.alias)

    def record_unload(self, current) -> None:
        self._stop_sampling()
        if self._load_id is not None:
            self._write("UPDATE loads SET unloaded_at = ?, peak_rss_mb = ? "
                        "WHERE id = ?",
                        (_now(), self._peak_rss or None, self._load_id))
        self._load_id = None

    def record_failure(self, alias: str, failure) -> None:
        """A load that did not come up. Takes the StartupFailed so the reason
        and the log path travel together rather than being re-derived."""
        self._write(
            """INSERT INTO loads (ts, alias, engine, ok, error, log_path)
               VALUES (?,?,?,0,?,?)""",
            (_now(), alias, "", getattr(failure, "reason", str(failure)),
             getattr(failure, "log_path", None)))

    # --------------------------------------------------------
    # Memory
    # --------------------------------------------------------
    def attach(self, supervisor) -> None:
        """The supervisor knows which pid owns the socket, which is the one
        holding the weights - `mtplx serve` re-execs, so it is not the process
        we spawned."""
        self._supervisor = supervisor

    def _start_sampling(self, alias: str) -> None:
        if self._supervisor is None:
            return
        self._stop_sampling()
        # A fresh event per sampler. Reusing one and clearing it here races the
        # old sampler: one that is mid-rss_mb() - an SSH call on a model host -
        # when the set() and clear() both happen never sees the stop, and keeps
        # writing the old alias's memory for good.
        stop = self._sampling = threading.Event()

        def loop():
            while not stop.wait(self.sample_interval):
                rss = self._supervisor.rss_mb()
                if rss is None:
                    continue
                self._peak_rss = max(self._peak_rss, rss)
                self._write("INSERT INTO memory (ts, alias, rss_mb) VALUES (?,?,?)",
                            (_now(), alias, round(rss, 1)))

        self._sampler = threading.Thread(target=loop, daemon=True)
        self._sampler.start()

    def _stop_sampling(self) -> None:
        self._sampling.set()
        self._sampler = None

    # --------------------------------------------------------
    # Reading
    # --------------------------------------------------------
    def per_model(self, since_hours: float | None = None) -> list[dict]:
        """One row per model.

        `avg_tps` is pooled - total tokens over total decode time - not the
        mean of per-request rates. min/max stay per-request extremes, which is
        what you want from them.

        TTFT is reported with its max alongside the mean because the worst case
        is the one a user notices, and a model with a good average and a 60s
        tail is exactly the failure the thinking-mode work in docs/ chased.
        """
        where = "WHERE ok = 1 AND preempted = 0"
        params: tuple = ()
        if since_hours:
            where += " AND ts >= ?"
            params = (datetime.fromtimestamp(
                time.time() - since_hours * 3600, timezone.utc
            ).isoformat(timespec="seconds"),)

        return self._query(f"""
            SELECT alias, engine,
                   COUNT(*)                       AS requests,
                   SUM(output_tokens)             AS output_tokens,
                   SUM(prompt_tokens)             AS prompt_tokens,
                   SUM(cached_tokens)             AS cached_tokens,
                   CASE WHEN SUM(decode_ms) > 0
                        THEN ROUND(SUM(output_tokens) * 1000.0 / SUM(decode_ms), 1)
                   END                            AS avg_tps,
                   ROUND(MIN(decode_tps), 1)      AS min_tps,
                   ROUND(MAX(decode_tps), 1)      AS max_tps,
                   ROUND(AVG(ttft_ms))            AS avg_ttft_ms,
                   ROUND(MIN(ttft_ms))            AS min_ttft_ms,
                   ROUND(MAX(ttft_ms))            AS max_ttft_ms,
                   ROUND(MAX(deferral_ms))        AS max_deferral_ms,
                   CASE WHEN SUM(mtp_drafted) > 0
                        THEN ROUND(SUM(mtp_accepted) * 1.0 / SUM(mtp_drafted), 3)
                   END                            AS mtp_accept,
                   SUM(mtp_drafted)               AS mtp_drafted,
                   ROUND(MAX(peak_rss_mb), 1)     AS peak_rss_mb,
                   MAX(rate_source)               AS rate_source
            FROM requests {where}
            GROUP BY alias, engine
            ORDER BY requests DESC""", params)

    def per_session(self, limit: int = 20) -> list[dict]:
        """One row per agent conversation, newest first.

        Pooled within the session for the same reason it is pooled per model -
        averaging per-turn rates over-weights the short turns, and an agent
        session is mostly short turns punctuated by a few long ones.

        `wall_s` is first turn to last, so it includes the time the client spent
        running tools between them. That gap is most of what a session feels
        like, and a decode rate that ignores it flatters every model equally.
        """
        return self._query("""
            SELECT session_id, alias, engine,
                   MAX(client)                         AS client,
                   COUNT(*)                            AS turns,
                   MIN(ts)                             AS started,
                   MAX(ts)                             AS ended,
                   SUM(output_tokens)                  AS output_tokens,
                   MAX(context_tokens)                 AS peak_context,
                   SUM(n_tool_calls)                   AS tool_calls,
                   CASE WHEN SUM(decode_ms) > 0
                        THEN ROUND(SUM(output_tokens) * 1000.0 / SUM(decode_ms), 1)
                   END                                 AS tps,
                   ROUND(MAX(ttft_ms))                 AS max_ttft_ms,
                   ROUND(MAX(deferral_ms))             AS max_deferral_ms,
                   ROUND(SUM(total_ms) / 1000.0, 1)    AS model_s,
                   SUM(preempted)                      AS preempted,
                   SUM(CASE WHEN finish_reason = 'length' THEN 1 ELSE 0 END)
                                                       AS truncated,
                   SUM(CASE WHEN ok = 0 THEN 1 ELSE 0 END) AS errors
            FROM requests
            WHERE session_id IS NOT NULL
            GROUP BY session_id
            ORDER BY MAX(id) DESC
            LIMIT ?""", (limit,))

    def session_turns(self, session_id: str) -> list[dict]:
        """Every turn in one session, oldest first - the shape of a run."""
        return self._query("""
            SELECT id, ts, alias, engine, requested, client_ip, client, ok, status,
                   preempted, output_tokens, prompt_tokens, cached_tokens,
                   context_tokens, decode_tps, rate_source, ttft_ms, ttfc_ms,
                   deferral_ms, total_ms, n_messages, n_tools, n_tool_calls,
                   tool_names, tool_args, finish_reason, mtp_accept, load_ms,
                   trace_file, trace_seq
            FROM requests WHERE session_id = ? ORDER BY id""", (session_id,))

    def session(self, session_id: str) -> dict | None:
        rows = [s for s in self.per_session(limit=1000)
                if s["session_id"] == session_id]
        return rows[0] if rows else None

    def turn(self, request_id: int) -> dict | None:
        """One turn's row, with its neighbours in the same session.

        `previous_messages` is what the turn before it sent, which is where
        this turn's new messages begin - the trace holds the whole conversation
        on every request, so that boundary is the difference between showing an
        exchange and showing a transcript.
        """
        rows = self._query("SELECT * FROM requests WHERE id = ?", (request_id,))
        if not rows:
            return None
        row = rows[0]

        neighbours = self._query(
            """SELECT id, n_messages,
                      LAG(id)  OVER (ORDER BY id) AS prev_id,
                      LEAD(id) OVER (ORDER BY id) AS next_id,
                      LAG(n_messages) OVER (ORDER BY id) AS prev_messages
               FROM requests WHERE session_id IS ? ORDER BY id""",
            (row["session_id"],))
        here = next((n for n in neighbours if n["id"] == request_id), {})
        row["prev_id"] = here.get("prev_id")
        row["next_id"] = here.get("next_id")
        row["previous_messages"] = here.get("prev_messages") or 0
        row["turn_number"] = next(
            (i + 1 for i, n in enumerate(neighbours) if n["id"] == request_id), None)
        row["session_turns"] = len(neighbours)
        return row

    def preempted_counts(self) -> dict[str, int]:
        rows = self._query("SELECT alias, COUNT(*) AS n FROM requests "
                           "WHERE preempted = 1 GROUP BY alias")
        return {r["alias"]: r["n"] for r in rows}

    def recent(self, limit: int = 40, offset: int = 0) -> list[dict]:
        return self._query(
            """SELECT id, ts, session_id, client_ip, client, alias, requested,
                      engine, ok, status,
                      preempted, context_tokens, output_tokens, decode_tps,
                      ttft_ms, ttfc_ms, total_ms, n_tool_calls, finish_reason,
                      rate_source, trace_seq
               FROM requests ORDER BY id DESC LIMIT ? OFFSET ?""",
            (limit, offset))

    def count_requests(self) -> int:
        rows = self._query("SELECT COUNT(*) AS n FROM requests")
        return rows[0]["n"] if rows else 0

    def memory_series(self, minutes: int = 30) -> list[dict]:
        stamp = datetime.fromtimestamp(
            time.time() - minutes * 60, timezone.utc).isoformat(timespec="seconds")
        return self._query(
            "SELECT ts, alias, rss_mb FROM memory WHERE ts >= ? ORDER BY ts",
            (stamp,))

    def loads(self, limit: int = 20) -> list[dict]:
        return self._query(
            "SELECT * FROM loads ORDER BY id DESC LIMIT ?", (limit,))

    # --------------------------------------------------------
    # The event log
    #
    # Deliberately not pruned. `memory` samples on a timer and needs
    # MEMORY_RETENTION_DAYS to stay small; events are written only when
    # something *changes* (see events.Events), so a week of hard use is tens of
    # rows. A cleaner here would only ever delete the history you opened the
    # page to read.
    # --------------------------------------------------------
    def record_event(self, kind: str, client: str | None,
                     detail: dict | None = None, text: str = "") -> None:
        self._write(
            "INSERT INTO events (ts, kind, client, text, detail) "
            "VALUES (?, ?, ?, ?, ?)",
            (_now(), kind, client, text,
             json.dumps(detail or {}, sort_keys=True)))

    def events(self, limit: int = 100) -> list[dict]:
        """Newest first, with `detail` already parsed.

        Load failures are not copied in here - they are already rows in
        `loads`, with their error and their log path. Merging the two is the
        dashboard's job, so a failure never goes stale in a second copy.
        """
        rows = self._query(
            "SELECT id, ts, kind, client, text, detail FROM events "
            "ORDER BY id DESC LIMIT ?", (limit,))
        for row in rows:
            try:
                row["detail"] = json.loads(row["detail"] or "{}")
            except (TypeError, ValueError):
                row["detail"] = {}
        return rows

    def drift(self) -> dict:
        """Per client, tools its set asks for that it has *stopped* offering.

        Narrower than `unknown` on purpose. A set may name a tool the client
        has simply never sent - a typo, or `task_complete`, which only some VS
        Code configurations offer - and that is not drift, it is an entry that
        has never applied. Reporting it would light the indicator permanently,
        which is the same as not having one.

        So a name counts only if some earlier event saw the client offering it.
        That is what a rename in a client release looks like from here, and it
        is the case where the model has silently lost a tool. The engine cannot
        detect it for a tool it has never seen, which is honest: on a fresh
        database there is nothing to have drifted from.

        Only the latest event per client decides. Fixing a set has to clear the
        indicator, or the Config tab reads as broken after you have dealt with
        it.
        """
        latest: dict = {}
        seen: dict = {}
        for row in self.events(limit=500):
            if row["kind"] != "tools":
                continue
            client = row["client"]
            if client not in latest:
                latest[client] = row["detail"]
            seen.setdefault(client, set()).update(
                row["detail"].get("matched") or [])

        out = {}
        for client, detail in latest.items():
            gone = [name for name in detail.get("unknown") or []
                    if name in seen.get(client, ())]
            if gone:
                out[client] = dict(detail, unknown=gone)
        return out

    def close(self) -> None:
        self._stop_sampling()
        try:
            with self._lock:
                self._db.close()
        except sqlite3.Error:
            pass
