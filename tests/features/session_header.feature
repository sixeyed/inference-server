Feature: Telling MTPLX which conversation a request belongs to
  MTPLX keeps one KV cache per session and extends it turn over turn. It reads
  the session from a header first, and only guesses from the prompt prefix
  when there is none. VS Code sends none it recognises, and the guess failed on
  11 September: once a prompt passed four times the shared system prompt, every
  turn became a new session holding its own copy of the context, and 25 turns
  took MTPLX from 36 GB to its 96 GiB cap.

  The engine already knows which conversation a turn belongs to, so it says so.

  Scenario: every turn of one conversation carries the same session id
    Given "qwen3.8" is loaded
    When a client sends 2 turns of one conversation
    Then both turns reached the upstream with the same session id

  Scenario: a new conversation gets a new session id
    Given "qwen3.8" is loaded
    When a client sends a long conversation and then starts a new one
    Then the two reached the upstream with different session ids

  Scenario: a session header the client sent is left alone
    Given "qwen3.8" is loaded
    When a client sends a turn with its own "X-Session-Id" of "client-chosen"
    Then the upstream received "X-Session-Id" as "client-chosen"
    And the upstream received no engine session id

  Scenario: collapsing a loop does not start a new session
    Given "qwen3.8" is loaded
    And VS Code collapses loops after 3 identical calls
    When VS Code sends a turn with 2 identical calls and then one with 5
    Then both turns reached the upstream with the same session id

  Scenario: overlapping turns of one conversation never share a session id
    Given "qwen3.8" is loaded and streams slowly
    When a second turn of a conversation arrives while its first is streaming
    Then the first turn carried a session id and the second did not
    And a later turn carries the session id again

  Scenario: a model on another engine gets no session id
    Given "kat-coder2.5" is loaded
    When a client sends 2 turns of one conversation
    Then no request reached the upstream with an engine session id

  Scenario: the trace records which headers the client sent, never their values
    Given "qwen3.8" is loaded with the API key "sk-do-not-record-me"
    When a client sends a turn with that API key and an interaction id
    Then the trace lists the client's header names
    And the trace does not contain the API key
    And the trace records the session id sent upstream
