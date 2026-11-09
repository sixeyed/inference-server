Tool sets shipped with the server. Each is a list of tool names, one per line,
naming the tools a client is allowed to offer the model. `#` starts a comment
anywhere on a line.

A set is an allowlist, not a patch: a name the client did not send is reported
rather than added, because the engine has no definition to add. Names are
matched against `tools[].function.name` exactly, and the client's own ordering
of the array is kept.

A set that matches nothing is treated as a typo and ignored - the client's
tools go through untouched. Zero tools does not make a coding agent fail, it
makes it quietly stop being an agent, which is worse than not filtering.

Sets are read per request, so editing one takes effect on the next turn without
restarting the engine.
