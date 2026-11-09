"""Grouping turns into sessions.

Nothing in an OpenAI request says which conversation it belongs to, so this is
inferred - and a wrong inference makes per-session numbers meaningless rather
than merely imprecise.
"""

from inference.sessions import SessionTracker, opening_key


def convo(turns, opening="system prompt", client="a"):
    return {"messages": [{"role": "system", "content": opening}]
            + [{"role": "user", "content": f"turn {i}"} for i in range(turns)]}


def test_a_growing_conversation_stays_one_session():
    t = SessionTracker()
    ids = [t.identify(convo(n)) for n in (0, 2, 5, 40, 200)]
    assert len(set(ids)) == 1, ids


def test_a_shorter_conversation_starts_a_new_one():
    # The signal that catches a restart whose opening is identical - observed
    # twice in the captured traces, as 19->3 and 405->3.
    t = SessionTracker()
    first = t.identify(convo(20))
    assert t.identify(convo(2)) != first


def test_a_different_opening_is_a_different_session():
    t = SessionTracker()
    assert t.identify(convo(3, "prompt A")) != t.identify(convo(3, "prompt B"))


def test_two_clients_are_never_merged():
    # Same opening from two machines is two sessions, not one interleaved mess.
    t = SessionTracker()
    a = t.identify(convo(3), client_ip="192.168.1.10")
    b = t.identify(convo(3), client_ip="192.168.1.20")
    assert a != b


def test_interleaved_sessions_keep_their_own_ids():
    t = SessionTracker()
    a1 = t.identify(convo(3, "A"))
    b1 = t.identify(convo(3, "B"))
    a2 = t.identify(convo(5, "A"))
    b2 = t.identify(convo(5, "B"))
    assert a1 == a2 and b1 == b2 and a1 != b1


def test_an_empty_request_does_not_raise():
    t = SessionTracker()
    assert t.identify({})
    assert t.identify({"messages": []})
    assert opening_key({}) == opening_key({"messages": []})


# ------------------------------------------------------------
# One request per session at a time
# ------------------------------------------------------------
# MTPLX answers a second request on a session that is still generating with a
# 409, not a queue - so the engine may only name a session upstream while no
# other request is using that name.

def test_a_session_can_be_claimed_once():
    t = SessionTracker()
    assert t.claim("abc-1") is True
    assert t.claim("abc-1") is False


def test_a_released_session_can_be_claimed_again():
    t = SessionTracker()
    t.claim("abc-1")
    t.release("abc-1")
    assert t.claim("abc-1") is True


def test_claims_are_per_session():
    t = SessionTracker()
    assert t.claim("abc-1") is True
    assert t.claim("abc-2") is True


def test_releasing_what_was_never_claimed_is_harmless():
    SessionTracker().release("never-claimed")
