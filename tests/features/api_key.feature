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

  Scenario: a real key is passed to the model server
    Given an engine with the API key "s3cret"
    When a client sends a chat request with the header "Bearer s3cret"
    Then the response is 200
    And the upstream saw the header "Bearer s3cret"

  Scenario: with a real key, a client that sends none is refused
    Given an engine with the API key "s3cret"
    When a client sends a chat request with no authorization header
    Then the response is 401
    And no model was started

  Scenario: with a real key, a wrong key is refused by the engine
    Given an engine with the API key "s3cret"
    When a client sends a chat request with the header "Bearer wrong"
    Then the response is 401
    And no model was started

  Scenario: Anthropic-API clients can send the key as x-api-key
    Given an engine with the API key "s3cret"
    When a client sends a chat request with x-api-key "s3cret"
    Then the response is 200
    And the upstream saw the header "Bearer s3cret"

  Scenario: with a real key, the dashboard's API needs it too
    Given an engine with the API key "s3cret" and its dashboard
    Then GET "/api/clients" without the key is 401
    And GET "/api/clients" with the key is 200
    And GET "/health" without the key is 200
    And GET "/" without the key is 200
    And POST "/api/unload" without the key is 401

  Scenario: a write from another site's page is refused
    Given an engine with its dashboard
    Then POST "/api/unload" from origin "http://evil.example" is 403
    And POST "/api/unload" from its own origin is 200
