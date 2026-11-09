Prompt files shipped with the server. Each is one coherent system prompt,
sectioned by the block it replaces in the client's own prompt.

`<preamble>` is the conventional name for the untagged opening of a client's
system message. Every other section replaces the tag of the same name, and only
the first occurrence of that tag - VS Code sends `<instructions>` twice, and the
second carries workspace skills and agents that must survive.

Blocks not named in a file are left exactly as the client sent them. In
particular `<toolUseInstructions>` and `<editFileInstructions>` describe the
contract the client parses replies against, so replacing those risks breaking
tool calling.
