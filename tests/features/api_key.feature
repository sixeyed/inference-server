Feature: The API key flows downstream
  There is one key for the whole system. The engine is started with it, launches
  the model server with it, and clients send it. It is passed through rather than
  substituted: if the client's key and the engine's disagree, we want the 401,
  not a silent success against a key the user didn't think they were using.

  Background:
    Given a model server that checks the API key

  Scenario: the client's key is passed through
    When a client sends a chat request with the header "Bearer none"
    Then the response is 200
    And the upstream saw the header "Bearer none"

  Scenario: a wrong key is not rescued by the engine
    When a client sends a chat request with the header "Bearer wrong"
    Then the response is 401

  Scenario: a client that sends no key at all still works
    When a client sends a chat request with no authorization header
    Then the response is 200
    And the upstream saw the header "Bearer none"

  Scenario: the readiness probe carries the key
    When the model is loaded from cold
    Then the engine reported it ready
