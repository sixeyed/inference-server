Feature: Rewriting the client's system prompt
  VS Code sends about 15,000 characters of Copilot framing on every turn, and a
  local model spends its context and its attention on it. Named blocks of that
  prompt can be substituted per client from the files shipped in
  inference/prompts/. The rule that matters: VS Code sends <instructions> twice
  and the second one carries the workspace's skills and agents, so only the
  first match of a tag is ever replaced.

  Background:
    Given "qwen3.8" is loaded

  Scenario: by default the client's prompt is forwarded untouched
    When VS Code sends its system prompt
    Then the upstream received the prompt unchanged

  Scenario: a configured block is substituted
    Given VS Code is configured to replace "instructions" with "terse-coder.md"
    When VS Code sends its system prompt
    Then the upstream prompt no longer contains "Prefer editing over rewriting"
    And the upstream prompt contains the replacement text
    And the trace records that "instructions" was rewritten
    And the trace keeps the prompt the client sent

  Scenario: the workspace's skills and agents survive
    Given VS Code is configured to replace "instructions" with "terse-coder.md"
    When VS Code sends its system prompt
    Then the upstream prompt still contains "<skills>"
    And the upstream prompt still contains "<agents>"

  Scenario: blocks that were not named are left alone
    Given VS Code is configured to replace "preamble" with "terse-coder.md"
    When VS Code sends its system prompt
    Then the upstream prompt still contains "Never mention tool names"
    And the upstream prompt still contains "Read the file before editing it"

  Scenario: the configuration is per client
    Given VS Code is configured to replace "preamble" with "terse-coder.md"
    When Cline sends the same system prompt
    Then the upstream received the prompt unchanged

  Scenario: the setting survives a restart
    Given VS Code is configured to replace "preamble" with "terse-coder.md"
    When the engine is restarted
    Then VS Code is still configured to replace "preamble"

  Scenario: a prompt file that is gone leaves the request alone
    Given VS Code is configured to replace "preamble" with "deleted.md"
    When VS Code sends its system prompt
    Then the upstream received the prompt unchanged

  Scenario: the dashboard offers only clients that have sent traffic
    When VS Code sends its system prompt
    Then the config page offers "vscode"
    And the config page lists "terse-coder.md" with its blocks
    And the config page flags the contract blocks

  Scenario: the dashboard can set and clear the substitution
    When VS Code sends its system prompt
    And the config page sets "vscode" to replace "preamble" with "terse-coder.md"
    Then VS Code's configured prompt file is "terse-coder.md"
    When the config page clears the prompt for "vscode"
    Then VS Code has no configured prompt
