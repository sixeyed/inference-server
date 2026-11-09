Feature: Starting and swapping models
  Only one model runs at a time. A request for a different one stops the current
  server immediately - anything in flight is cut, because changing model means
  the old model's answer is no longer wanted.

  Scenario: a cold request starts the model and is held open until it is ready
    Given nothing is loaded
    When a client sends a chat request for "qwen3.8"
    Then the response is 200
    And "qwen3.8" was started
    And the engine reports "qwen3.8" as loaded

  Scenario: asking for the same model again does not restart it
    Given "qwen3.8" is loaded
    When a client sends a chat request for "qwen3.8"
    Then the model was started 1 time

  Scenario: asking for a different model swaps
    Given "qwen3.8" is loaded
    When a client sends a chat request for "kat-coder2.5"
    Then the response is 200
    And the engine reports "kat-coder2.5" as loaded
    And the previous server was stopped

  Scenario: a model that dies during startup reports why
    Given the model server will fail to start with "error: unknown flag --nope"
    When a client sends a chat request for "qwen3.8"
    Then the response is 503
    And the error message contains "error: unknown flag --nope"
    And the error message names the log file

  Scenario: a repeated request for a failing model does not relaunch it
    Given the model server will fail to start with "boom"
    When a client sends a chat request for "qwen3.8"
    And a client sends a chat request for "qwen3.8"
    Then the model was started 1 time
