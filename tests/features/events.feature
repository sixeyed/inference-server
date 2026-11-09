Feature: The engine's event log
  What the engine changed about a request, and when. This used to be a console
  line printed once per engine run, guarded by a boolean that was set on the
  first occurrence and never cleared - so it reported the first state and hid
  every later one. The case it hid is the one worth seeing: a client release
  renaming a tool halfway through a session, which the set then silently drops.

  Drift is narrower than "the set names something the client did not send". A
  set entry the client has never offered has simply never applied; reporting it
  would light the indicator permanently. Drift is a tool the client *did* offer
  and has stopped offering.

  Background:
    Given "qwen3.8" is loaded

  Scenario: nothing is recorded when the engine changes nothing
    When VS Code sends a request offering its tools
    Then the log is empty

  Scenario: filtering the tools is recorded once
    Given VS Code is configured to use the "vscode-core.txt" tool set
    When VS Code sends a request offering its tools
    And VS Code sends a request offering its tools
    Then the log has one "tools" entry

  Scenario: a set entry the client has never sent is not drift
    Given VS Code is configured to use the "vscode-core.txt" tool set
    When VS Code sends a request offering its tools
    Then the config page reports no drift

  Scenario: a tool the client stops offering is drift
    Given VS Code is configured to use the "vscode-core.txt" tool set
    When VS Code sends a request offering its tools
    And a client release renames "read_file"
    And VS Code sends a request offering its tools
    Then the config page reports drift for "vscode" naming "read_file"
    And the log reports "read_file" as no longer offered

  Scenario: drift that appears mid-session is not swallowed
    Given VS Code is configured to use the "vscode-core.txt" tool set
    When VS Code sends a request offering its tools
    And a client release renames "read_file"
    And VS Code sends a request offering its tools
    Then the log has two "tools" entries

  Scenario: the client going back to the old name clears the flag
    Given VS Code is configured to use the "vscode-core.txt" tool set
    When VS Code sends a request offering its tools
    And a client release renames "read_file"
    And VS Code sends a request offering its tools
    And the release is rolled back
    And VS Code sends a request offering its tools
    Then the config page reports no drift
