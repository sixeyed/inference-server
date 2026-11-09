Feature: Filtering the client's tool list
  VS Code offers its whole tool surface on every turn - 80 definitions, about
  23,000 tokens of JSON schema. Across 15 captured trace files it offered 103
  distinct tools and the model called 18. A tool set names the ones a client is
  allowed to offer, per client, so the rest stop costing context and stop being
  wrong answers the model can pick. The rule that matters: a set that matches
  nothing is a typo, and a coding agent handed zero tools fails silently.

  Background:
    Given "qwen3.8" is loaded

  Scenario: by default every tool the client sent is forwarded
    When VS Code sends a request offering its tools
    Then the upstream received all of the tools

  Scenario: a configured set cuts the tools down
    Given VS Code is configured to use the "vscode-core.txt" tool set
    When VS Code sends a request offering its tools
    Then the upstream received the tool "read_file"
    And the upstream did not receive the tool "run_notebook_cell"
    And the trace records that "run_notebook_cell" was dropped
    And the trace keeps all of the tools the client sent

  Scenario: the configuration is per client
    Given VS Code is configured to use the "vscode-core.txt" tool set
    When Cline sends a request offering its tools
    Then the upstream received all of the tools

  Scenario: filtering takes effect without a restart
    When VS Code sends a request offering its tools
    And VS Code is configured to use the "vscode-core.txt" tool set
    And VS Code sends a request offering its tools
    Then the upstream did not receive the tool "run_notebook_cell"

  Scenario: the setting survives a restart
    Given VS Code is configured to use the "vscode-core.txt" tool set
    When the engine is restarted
    Then VS Code's tool set is still "vscode-core.txt"

  Scenario: a tool set file that is gone leaves the request alone
    Given VS Code is configured to use the "deleted.txt" tool set
    When VS Code sends a request offering its tools
    Then the upstream received all of the tools

  Scenario: a set that matches nothing never sends a toolless request
    Given a tool set "typo.txt" whose every name is a typo
    And VS Code is configured to use the "typo.txt" tool set
    When VS Code sends a request offering its tools
    Then the upstream received all of the tools

  Scenario: a prompt and a tool set are configured independently
    Given VS Code is configured to replace "preamble" with "terse-coder.md"
    And VS Code is configured to use the "vscode-core.txt" tool set
    When the prompt for "vscode" is cleared
    Then VS Code's tool set is still "vscode-core.txt"

  Scenario: the dashboard can set and clear the tool set
    When VS Code sends a request offering its tools
    And the config page sets "vscode" to use "vscode-core.txt"
    Then VS Code's tool set is still "vscode-core.txt"
    When the config page clears the tool set for "vscode"
    Then VS Code has no configured tool set
