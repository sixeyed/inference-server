Feature: Cutting a turn to swap models
  A swap kills whatever is streaming. That has to be told apart from a server
  crash - both look like the upstream vanishing mid-response, and a log that
  conflates them makes the next real crash invisible.

  Background:
    Given "qwen3.8" is loaded and answering slowly

  Scenario: switching model cuts the turn in flight
    When a streaming request is in flight
    And a client sends a chat request for "kat-coder2.5"
    Then the in-flight response is cut short
    And the swap did not wait for it
    And the engine reports "kat-coder2.5" as loaded
    And the engine is not in a failed state

  Scenario: the cut turn is recorded as preempted, not as an error
    When a streaming request is in flight
    And a client sends a chat request for "kat-coder2.5"
    Then one request is recorded as preempted
    And the preempted request is absent from the per-model statistics
    And the preempted request still appears in the recent requests
