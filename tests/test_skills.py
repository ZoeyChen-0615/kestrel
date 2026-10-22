from kestrel.skills import SkillCatalog


def write_skill(root, name, body="Do the thing.", frontmatter=""):
    folder = root / name
    folder.mkdir(parents=True)
    (folder / "SKILL.md").write_text(frontmatter + body, encoding="utf-8")
    return folder


def test_frontmatter_and_index(tmp_path):
    write_skill(tmp_path, "weather", "Use curl.", "---\nname: weather\ndescription: Look up the weather\n---\n")
    catalog = SkillCatalog([("bundled", tmp_path)])
    assert catalog.index_text() == "- weather: Look up the weather"


def test_directory_name_is_the_default_skill_name(tmp_path):
    write_skill(tmp_path, "plain", "Just instructions.")
    assert "plain" in SkillCatalog([("a", tmp_path)]).scan()


def test_later_roots_override_earlier_ones(tmp_path):
    low, high = tmp_path / "low", tmp_path / "high"
    write_skill(low, "s", "LOW")
    write_skill(high, "s", "HIGH")
    skill = SkillCatalog([("bundled", low), ("workspace", high)]).scan()["s"]
    assert skill.body == "HIGH" and skill.source == "workspace"


def test_missing_requirements_mark_a_skill_unavailable(tmp_path):
    write_skill(tmp_path, "needs", "x", "---\ndescription: needs a tool\nrequires:\n  bins: [definitely-not-installed-xyz]\n  env: [KESTREL_TEST_MISSING_VAR]\n---\n")
    catalog = SkillCatalog([("a", tmp_path)])
    skill = catalog.scan()["needs"]
    assert not skill.available and "bin:definitely-not-installed-xyz" in skill.missing
    assert "unavailable" in catalog.index_text()
    result = catalog.load("needs")
    assert result.is_error and "unavailable" in result.text


def test_load_returns_body_and_base_directory(tmp_path):
    folder = write_skill(tmp_path, "dog", "Read references/care.md first.")
    result = SkillCatalog([("a", tmp_path)]).load("dog")
    assert not result.is_error and str(folder) in result.text and "references/care.md" in result.text


def test_unknown_skill_lists_alternatives_and_broken_files_are_skipped(tmp_path):
    write_skill(tmp_path, "good", "ok")
    (tmp_path / "empty").mkdir()
    (tmp_path / "empty" / "SKILL.md").write_text("---\nname: empty\n---\n", encoding="utf-8")  # no body
    result = SkillCatalog([("a", tmp_path)]).load("nope")
    assert result.is_error and "good" in result.text and "empty" not in result.text
