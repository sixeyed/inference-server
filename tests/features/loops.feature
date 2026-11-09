Feature: Breaking a stuck tool-call loop
  A model that gets stuck re-issuing one tool call cannot sample its way out.
  The repetition is across turns, not within a generation, so temperature and
  the repetition penalties never see it - replaying a real stuck turn produced
  byte-identical output six times out of six with the model's own sampling.

  The engine is the only component that can see it: chat completions is
  stateless, so the whole history arrives every turn, and the client did not
  stop it after thirty-seven tries. Replacing the run with a note that carries
  the result but never the arguments was measured to break the loop 4 times in
  4; keeping one copy of the call broke it 0 times in 3.

  Background:
    Given "qwen3.8" is loaded

  Scenario: by default a repeated call is forwarded untouched
    When VS Code sends a history with 6 identical tool calls
    Then the upstream received the whole history

  Scenario: a run past the trigger count is collapsed
    Given VS Code collapses loops after 3 identical calls
    When VS Code sends a history with 6 identical tool calls
    Then the upstream history is shorter than the client sent
    And the upstream no longer contains the repeated command
    And the upstream still contains what the command returned
    And the trace records that 6 calls were collapsed
    And the trace keeps the whole history the client sent

  Scenario: a run below the trigger count is left alone
    Given VS Code collapses loops after 8 identical calls
    When VS Code sends a history with 6 identical tool calls
    Then the upstream received the whole history

  Scenario: a poll whose output changes is never collapsed
    Given VS Code collapses loops after 3 identical calls
    When VS Code sends a history with 6 polls that each report progress
    Then the upstream received the whole history

  Scenario: turning it off leaves a genuine poll alone
    Given VS Code collapses loops after 3 identical calls
    And VS Code collapses loops after 0 identical calls
    When VS Code sends a history with 6 identical tool calls
    Then the upstream received the whole history

  Scenario: the configuration is per client
    Given VS Code collapses loops after 3 identical calls
    When Cline sends a history with 6 identical tool calls
    Then the upstream received the whole history

  Scenario: the setting survives a restart
    Given VS Code collapses loops after 3 identical calls
    When the engine is restarted
    Then VS Code still collapses loops after 3

  Scenario: the intervention is on the Logs page
    Given VS Code collapses loops after 3 identical calls
    When VS Code sends a history with 6 identical tool calls
    Then the log has a "loop" entry

  Scenario: the config page sets and clears the trigger count
    When VS Code sends a history with 6 identical tool calls
    And the config page sets "vscode" to collapse after 4
    Then VS Code still collapses loops after 4
    When the config page sets "vscode" to collapse after 0
    Then VS Code has loop collapsing off
