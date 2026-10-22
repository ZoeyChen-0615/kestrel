import pytest

from kestrel.common import ConflictError
from kestrel.store import Store


@pytest.fixture
def store(tmp_path):
    s = Store(tmp_path / "db.sqlite")
    yield s
    s.close()


def test_sequence_numbers_are_monotonic_per_session(store):
    store.create_session(session_id="a")
    store.create_session(session_id="b")
    assert [store.append("a", {"n": i}) for i in range(3)] == [1, 2, 3]
    assert store.append("b", {"n": 0}) == 1
    assert store.head("a") == 3


def test_expected_head_detects_a_competing_writer(store):
    store.create_session(session_id="a")
    store.append("a", {"x": 1})
    with pytest.raises(ConflictError):
        store.append("a", {"x": 2}, expected_head=0)
    assert store.append("a", {"x": 2}, expected_head=1) == 2


def test_summary_is_a_projection_and_never_deletes_rows(store):
    store.create_session(session_id="s")
    for i in range(1, 5):
        store.append("s", {"role": "user", "content": f"m{i}"})
    store.append("s", {"content": "SUMMARY", "covers_through": 2}, kind="summary")
    summary, covers, messages = store.load_context("s")
    assert summary == "SUMMARY" and covers == 2
    assert [m["content"] for _, m in messages] == ["m3", "m4"]
    assert len(store.rows("s")) == 5  # originals still present


def test_kv_compare_and_swap(store):
    assert store.kv_put("ns", "k", {"v": 1}, expected_version=None) == 1
    with pytest.raises(ConflictError):
        store.kv_put("ns", "k", {"v": 2}, expected_version=None)  # already exists
    with pytest.raises(ConflictError):
        store.kv_put("ns", "k", {"v": 2}, expected_version=7)  # stale version
    assert store.kv_put("ns", "k", {"v": 2}, expected_version=1) == 2
    assert store.kv_get("ns", "k") == (2, {"v": 2})


def test_episodes_are_unique_per_user_message(store):
    assert store.add_episode("s", 1, "t", "m", 2, b"\x00" * 8) is True
    assert store.add_episode("s", 1, "t", "m", 2, b"\x00" * 8) is False
