Feature: Relaying faithfully
  The engine is a transparent relay except in two places, and both are recorded
  so a trace never lies about what the client actually sent.

  Scenario: empty tool-call arguments are repaired before they poison the session
    Given "qwen3.8" is loaded
    When a client sends a conversation containing a tool call with empty arguments
    Then the upstream received "{}" as the arguments
    And the trace records the repair
    And the trace keeps the untouched original

  Scenario: the model field is rewritten when the upstream expects another name
    Given "gemma-4-31b" is loaded with deprecated models served
    When a client sends a chat request for "gemma-4-31b"
    Then the upstream received the model "FakeRockert543/gemma-4-31b-it-MLX-8bit"
    And the trace records the rewrite

  Scenario: an alias the upstream already answers to is not rewritten
    Given "qwen3.8" is loaded
    When a client sends a chat request for "qwen3.8"
    Then the trace records no rewrite

  Scenario: a client that hangs up mid-stream is recorded as a disconnect
    Given "qwen3.8" is loaded and streams slowly
    When a client hangs up after the first chunk
    Then the trace records that the client disconnected

  Scenario: reasoning before content shows up as deferral
    Given "qwen3.8" is loaded and thinks before answering
    When a client sends a streaming chat request
    Then time-to-first-token is earlier than time-to-first-content
    And the recorded deferral is greater than zero

  Scenario: an Anthropic-API request for an OpenAI-only model is refused before any swap
    Given "qwen3.8" is loaded
    When a client sends a /v1/messages request for "kat-coder2.5"
    Then the response is 404 "unsupported_api"
    And "qwen3.8" is still the only model started

  Scenario: a model field that is not a string is served, not a dropped connection
    Given "qwen3.8" is loaded
    When a client sends a chat request whose model is 42
    Then the response is 200

  Scenario: null tool-call arguments are repaired, not a dropped connection
    Given "qwen3.8" is loaded
    When a client sends a conversation containing a tool call with null arguments
    Then the upstream received "{}" as the arguments

  Scenario: a GET relay never brings back the model it saw before a swap
    Given "qwen3.8" is loaded
    When the engine swaps to "kat-coder2.5" while a GET still sees "qwen3.8" ready
    Then the models started were "qwen3.8, kat-coder2.5"
    And the engine is serving "kat-coder2.5"
