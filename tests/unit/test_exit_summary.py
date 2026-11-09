"""The line the engine prints on the way out.

This ran only in the shutdown `finally` of `inference engine`, so a NameError in
it stayed invisible until someone stopped the server - and by then the model was
already down, so nothing looked wrong except a traceback in a log nobody reads.
It shipped with `os` unimported for exactly that reason.
"""
import types

from inference import cli


def _tracer(tmp_path, turns=3, content=b"x" * 2048):
    path = tmp_path / "trace.jsonl"
    path.write_bytes(content)
    return types.SimpleNamespace(path=str(path), count=turns)


def test_it_reports_the_turns_and_the_size(tmp_path):
    out = cli.trace_summary(_tracer(tmp_path), no_trace=False)
    assert "Captured 3 turns" in out
    assert str(tmp_path / "trace.jsonl") in out
    assert "0.0MB" in out


def test_it_says_nothing_when_tracing_is_off(tmp_path):
    assert cli.trace_summary(_tracer(tmp_path), no_trace=True) == ""


def test_it_says_nothing_when_no_trace_was_written(tmp_path):
    """--no-trace off but the engine served nothing, so the file was never
    created. Stat-ing it would raise inside the finally."""
    tracer = types.SimpleNamespace(path=str(tmp_path / "absent.jsonl"), count=0)
    assert cli.trace_summary(tracer, no_trace=False) == ""


def test_no_module_reaches_for_a_global_that_is_not_there():
    """The general form of the bug: a function referencing a module-level name
    that was never imported. It only shows up when that line runs, which for
    anything in a shutdown path can be weeks.

    `symtable` does real scope analysis, so this distinguishes a missing import
    from a local variable that merely shares its name - the thing a hand-written
    list of names gets wrong. The repo is stdlib-only by policy, so this stands
    in for the linter that would otherwise catch it.
    """
    import builtins
    import importlib
    import pkgutil
    import symtable

    import inference

    missing = []
    for info in pkgutil.walk_packages(inference.__path__, "inference."):
        module = importlib.import_module(info.name)
        source = open(module.__file__, encoding="utf-8").read()
        top = symtable.symtable(source, module.__file__, "exec")

        def walk(table, path):
            for sym in table.get_symbols():
                if not (sym.is_global() and sym.is_referenced()):
                    continue
                name = sym.get_name()
                if hasattr(module, name) or hasattr(builtins, name):
                    continue
                missing.append(f"{info.name}:{path} -> {name}")
            for child in table.get_children():
                walk(child, f"{path}.{child.get_name()}" if path else child.get_name())

        walk(top, "")
    assert not missing, "names used but never imported:\n  " + "\n  ".join(missing)
