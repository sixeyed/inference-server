Feature: Working out which model a client meant
  The `model` field is the only routing signal there is, and clients send the
  same model under several legitimate names. Getting this wrong doesn't error -
  it quietly benchmarks the wrong model - so the mapping is spelled out.

  Scenario Outline: the client can say the model's name several ways
    When a client asks for "<sent>"
    Then it resolves to "<alias>"

    Examples:
      | sent                          | alias        |
      | kat-coder2.5                  | kat-coder2.5 |
      | kat-coder                     | kat-coder2.5 |
      | KAT-CODER2.5                  | kat-coder2.5 |
      | local/qwen3.8                 | qwen3.8      |
      | qwen3.8                       | qwen3.8      |
      | unsloth/Qwen3.6-35B-A3B-GGUF  | qwen3.6-moe  |

  Scenario Outline: names that must not resolve
    When a client asks for "<sent>"
    Then it does not resolve

    Examples:
      | sent          |
      | gpt-4o        |
      | gemma-4-31b   |
      | qwen3.6       |
      |               |

  Scenario: deprecated models are routable only when asked for
    Given the deprecated models are included
    When a client asks for "gemma-4-31b"
    Then it resolves to "gemma-4-31b"

  Scenario: the model list offers every live model, not just the loaded one
    Given the engine is running
    Then the model list contains "qwen3.8"
    And the model list contains "kat-coder2.5"
    And the model list does not contain "gemma-4-31b"
    And no model is loaded

  Scenario: a test-only model routes by name but is never offered to clients
    Given the engine is running
    When a client asks for "qwen3-tiny"
    Then it resolves to "qwen3-tiny"
    And the model list does not contain "qwen3-tiny"
