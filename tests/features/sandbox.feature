Feature: Fetching weights before a sandboxed launch
  A sandboxed server cannot download weights - the egress sandbox denies the
  socket. So missing weights are pulled first, outside the sandbox, by the
  engine's own code, while the model already loaded carries on serving. The
  server is then launched sandboxed and never needs the network.

  With pulling turned off, a launch that cannot work is refused before
  anything is stopped: finding out afterwards costs a working session to learn
  something we could have known.

  Scenario: missing weights are pulled, then the model is launched
    Given "qwen3.8" is loaded
    And "muse-glimmer" has no weights on disk
    When a client sends a chat request for "muse-glimmer"
    Then the response is 200
    And the weights for "muse-glimmer" were pulled

  Scenario: weights already on disk are not pulled again
    Given "qwen3.8" is loaded
    When a client sends a chat request for "muse-glimmer"
    Then the response is 200
    And nothing was pulled

  Scenario: a pull that fails keeps the running model
    Given "qwen3.8" is loaded
    And "muse-glimmer" has no weights on disk
    And pulling fails with "HuggingFace answered 503"
    When a client sends a chat request for "muse-glimmer"
    Then the response is 503
    And the error says the weights could not be fetched
    And "qwen3.8" is still serving
    And nothing was stopped

  Scenario: with pulling off, a model with no weights is refused before the running one is killed
    Given "qwen3.8" is loaded with pulling off
    And "muse-glimmer" has no weights on disk
    When a client sends a chat request for "muse-glimmer"
    Then the response is 503
    And the error explains that the sandbox blocks the download
    And the error says how to fetch the weights
    And "qwen3.8" is still serving
    And nothing was stopped

  Scenario: with the sandbox off the same launch is attempted
    Given the sandbox is disabled
    And "muse-glimmer" has no weights on disk
    When a client sends a chat request for "muse-glimmer"
    Then the response is 200

  Scenario: a failed load is recorded, not just logged
    Given "qwen3.8" is loaded with pulling off
    And "muse-glimmer" has no weights on disk
    When a client sends a chat request for "muse-glimmer"
    Then a failed load is recorded for "muse-glimmer"
