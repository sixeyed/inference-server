Feature: Stripping client parameters
  Clients send their own sampling settings, which override whatever the model's
  file tuned it to - VS Code sends temperature 0.1 on every request. Stripping
  the key hands the decision back to the model file, and it has to be reachable
  mid-incident: no restart, and still set after one.

  Background:
    Given "qwen3.8" is loaded

  Scenario: by default the client's parameters are passed through untouched
    When a client sends a chat request with temperature 0.1
    Then the upstream received a temperature of 0.1

  Scenario: a stripped parameter never reaches the model
    Given "temperature" is stripped
    When a client sends a chat request with temperature 0.1
    Then the upstream received no temperature
    And the trace records that "temperature" was stripped
    And the trace keeps the temperature the client sent

  Scenario: stripping takes effect without a restart
    When a client sends a chat request with temperature 0.1
    And "temperature" is stripped
    And a client sends a chat request with temperature 0.1
    Then the upstream received no temperature

  Scenario: the setting survives a restart
    Given "temperature" is stripped
    When the engine is restarted
    Then "temperature" is still stripped

  Scenario Outline: the request itself is never stripped
    Given "<key>" is stripped
    Then "<key>" is not in the strip list

    Examples:
      | key         |
      | model       |
      | messages    |
      | tools       |
      | stream      |
      | tool_choice |
