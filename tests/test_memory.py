import pytest

from kestrel.llm import HashingEmbedder, ScriptedChat, text_reply
from kestrel.memory import EpisodicMemory, LongTermMemory
from kestrel.memory.bm25 import bm25_scores, tokenize
from kestrel.memory.longterm import parse_operations
from kestrel.store import Store


def test_tokenizer_handles_latin_and_cjk():
    assert tokenize("Hello, World_1!") == ["hello", "world_1"]
    assert tokenize("小狗荷包蛋") == list("小狗荷包蛋") + ["小狗", "狗荷", "荷包", "包蛋"]


def test_stopwords_are_ignored_so_function_word_overlap_scores_zero():
    docs = ["my dog is called Mochi and he is a corgi", "the weather is nice today"]
    assert bm25_scores("what is the capital of Japan", docs) == [0.0, 0.0]
    assert tokenize("What is the capital of Japan?") == ["capital", "japan"]
    assert bm25_scores("tell me about Mochi", docs)[0] > 0


def test_bm25_prefers_documents_with_the_rare_term():
    docs = ["the cat sat", "the dog barked loudly", "error code ZX4471 occurred"]
    scores = bm25_scores("ZX4471", docs)
    assert scores[2] > 0 and scores[0] == scores[1] == 0


@pytest.fixture
def episodic():
    store = Store(":memory:")
    return EpisodicMemory(store, HashingEmbedder(), min_dense=0.2), store


async def test_hybrid_search_finds_the_relevant_episode(episodic):
    memory, _ = episodic
    await memory.learn("s", 1, "my dog is called Mochi and loves pumpkin", "Nice, Mochi sounds lovely.")
    await memory.learn("s", 3, "how do I configure nginx reverse proxy", "Use proxy_pass inside a location block.")
    await memory.learn("s", 5, "what's the capital of France", "Paris.")
    hits = await memory.search("what is my dog's name", k=3)
    assert hits and "Mochi" in hits[0].text


async def test_exact_rare_terms_are_found_through_the_sparse_channel(episodic):
    memory, _ = episodic
    await memory.learn("s", 1, "ticket ZX4471 is blocked on legal review", "Noted.")
    await memory.learn("s", 3, "we talked about holiday plans in spain", "Sounds fun.")
    hits = await memory.search("ZX4471", k=2)
    assert hits[0].sparse > 0 and "ZX4471" in hits[0].text


async def test_exclude_hides_exchanges_that_are_still_in_context(episodic):
    memory, _ = episodic
    await memory.learn("cur", 1, "my dog is called Mochi", "ok")
    await memory.learn("old", 1, "my dog is called Mochi", "ok")
    hits = await memory.search("dog Mochi", k=5, exclude=lambda session, seq: session == "cur")
    assert [h.session_id for h in hits] == ["old"]


async def test_learning_is_idempotent_and_search_on_empty_store_is_safe(episodic):
    memory, store = episodic
    assert await memory.search("anything") == []
    assert await memory.learn("s", 1, "a", "b") is True
    assert await memory.learn("s", 1, "a", "b") is False
    assert len(store.episodes()) == 1


async def test_episodes_from_a_different_embedding_model_are_ignored(episodic):
    memory, store = episodic
    store.add_episode("s", 1, "dog Mochi", "some-other-model", 4, b"\x00" * 16)
    assert await memory.search("dog Mochi") == []


# ---- long-term memory --------------------------------------------------------

def test_long_term_add_dedupe_remove(tmp_path):
    memory = LongTermMemory(tmp_path / "MEMORY.md")
    assert memory.add("User lives in Chicago") == "added"
    assert memory.add("  user lives   in chicago ") == "duplicate"
    assert memory.add("x" * 500).startswith("fact must be")
    assert memory.facts() == ["User lives in Chicago"]
    assert memory.remove("chicago") == 1 and memory.facts() == []


def test_long_term_quota_drops_oldest_first(tmp_path):
    memory = LongTermMemory(tmp_path / "MEMORY.md", max_chars=120)
    for i in range(10):
        memory.add(f"fact number {i} with some padding text")
    facts = memory.facts()
    assert len("\n".join(facts)) <= 120 and facts[-1].startswith("fact number 9") and "fact number 0" not in " ".join(facts)


def test_parse_operations_tolerates_fences_and_prose():
    text = 'Sure!\n```json\n[{"op": "add", "text": "Likes tea"}, "junk"]\n```'
    assert parse_operations(text) == [{"op": "add", "text": "Likes tea"}]
    assert parse_operations("no json here") == []
    assert parse_operations("[not json]") == []


def test_apply_operations_ignores_malformed_items(tmp_path):
    memory = LongTermMemory(tmp_path / "m.md")
    changed = memory.apply_operations([{"op": "add", "text": "Has a dog"}, {"op": "add"}, {"op": "explode", "text": "x"}, {"op": "remove", "text": "nothing"}])
    assert changed == 1 and memory.facts() == ["Has a dog"]


async def test_extract_from_applies_model_operations(tmp_path):
    memory = LongTermMemory(tmp_path / "m.md")
    llm = ScriptedChat([text_reply('[{"op":"add","text":"User is learning Rust"}]')])
    assert await memory.extract_from(llm, "user: I started learning Rust") == 1
    assert memory.facts() == ["User is learning Rust"]
    assert "Existing memory" in llm.requests[0]["messages"][1]["content"]


async def test_unrelated_queries_that_only_share_common_words_return_nothing():
    """Regression: 'is'/'the'/'what' overlap used to admit irrelevant memories through the keyword channel."""
    store = Store(":memory:")
    memory = EpisodicMemory(store, HashingEmbedder(), min_dense=0.99)   # isolate the sparse channel
    await memory.learn("a", 1, "My dog is called Mochi and he is a corgi", "What a lovely dog.")
    assert await memory.search("What is the capital of Japan?", k=5) == []
    assert [h.session_id for h in await memory.search("tell me more about Mochi", k=5)] == ["a"]


def test_replace_swaps_exactly_one_fact_in_place(tmp_path):
    memory = LongTermMemory(tmp_path / "m.md")
    for fact in ("Name is Priya", "Lives in Chicago", "Allergic to peanuts"):
        memory.add(fact)
    assert memory.replace("chicago", "Lives in Boston") == "replaced"
    assert memory.facts() == ["Name is Priya", "Lives in Boston", "Allergic to peanuts"]   # order kept, others untouched


def test_replace_refuses_ambiguity_and_adds_when_nothing_matches(tmp_path):
    memory = LongTermMemory(tmp_path / "m.md")
    memory.add("Likes green tea")
    memory.add("Likes black tea")
    assert memory.replace("tea", "Likes coffee") == "ambiguous" and len(memory.facts()) == 2
    assert memory.replace("whisky", "Likes whisky") == "added" and "Likes whisky" in memory.facts()
    assert memory.replace("whisky", "x" * 500).startswith("fact must be")


def test_exact_removal_does_not_touch_facts_that_merely_contain_the_text(tmp_path):
    memory = LongTermMemory(tmp_path / "m.md")
    memory.add("Lives in Boston (moved from Chicago)")
    memory.add("Lives in Boston")
    assert memory.remove("Lives in Boston", exact=True) == 1
    assert memory.facts() == ["Lives in Boston (moved from Chicago)"]
