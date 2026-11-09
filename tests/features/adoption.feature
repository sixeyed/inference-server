Feature: Picking up a server that was already running
  A server started by hand with `inference run` belongs to whoever started it. The engine
  uses it rather than restarting it, and never kills it on the way out - the
  first health probe a client sends should not execute somebody's work.

  Scenario: a server already on the port is adopted
    Given a model server is already serving "qwen3.8"
    When the engine starts
    Then the engine reports "qwen3.8" as loaded
    And it is marked as adopted
    And nothing was started

  Scenario: an adopted server is used, not relaunched
    Given a model server is already serving "qwen3.8"
    And the engine has adopted it
    When a client sends a chat request for "qwen3.8"
    Then the response is 200
    And nothing was started

  Scenario: an adopted server survives engine shutdown
    Given a model server is already serving "qwen3.8"
    And the engine has adopted it
    When the engine shuts down
    Then the model server is still running

  Scenario: a server serving something unrecognised is still left alone
    Given a model server is already serving "some-other-build"
    When the engine starts
    Then it is marked as adopted
    And nothing was started
