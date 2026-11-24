import hashlib

from indexer import file_hash, plan_changes, scan_folder


def test_file_hash_matches_sha256(tmp_path):
    note = tmp_path / "a.md"
    note.write_bytes(b"hello")
    assert file_hash(note) == hashlib.sha256(b"hello").hexdigest()


def test_scan_folder_keeps_only_notes(tmp_path, monkeypatch):
    (tmp_path / "a.md").write_text("one", encoding="utf-8")
    (tmp_path / "B.PDF").write_bytes(b"%PDF")
    (tmp_path / "c.txt").write_text("ignored", encoding="utf-8")
    (tmp_path / "sub").mkdir()
    monkeypatch.setenv("STORAGE_DIR", str(tmp_path))
    scan = scan_folder()
    assert sorted(scan) == ["B.PDF", "a.md"]
    assert scan["a.md"] == file_hash(tmp_path / "a.md")


def test_plan_changes_all_cases():
    disk = {"a.md": "h1", "b.md": "h2", "c.md": "h3"}
    registry = {"b.md": "h2", "c.md": "OLD", "d.md": "h4"}
    new, modified, deleted = plan_changes(disk, registry)
    assert new == ["a.md"]
    assert modified == ["c.md"]
    assert deleted == ["d.md"]


def test_plan_changes_nothing_changed():
    same = {"a.md": "h1", "b.md": "h2"}
    assert plan_changes(same, dict(same)) == ([], [], [])


def test_plan_changes_first_run_everything_is_new():
    disk = {"a.md": "h1", "b.md": "h2"}
    assert plan_changes(disk, {}) == (["a.md", "b.md"], [], [])


def test_plan_changes_empty_folder_deletes_everything():
    registry = {"a.md": "h1", "b.md": "h2"}
    assert plan_changes({}, registry) == ([], [], ["a.md", "b.md"])


def test_plan_changes_does_not_modify_its_inputs():
    disk = {"a.md": "h1"}
    registry = {"b.md": "h2"}
    plan_changes(disk, registry)
    assert disk == {"a.md": "h1"} and registry == {"b.md": "h2"}


def test_rename_is_one_deletion_plus_one_addition():
    disk = {"new_name.md": "h1"}
    registry = {"old_name.md": "h1"}
    assert plan_changes(disk, registry) == (["new_name.md"], [], ["old_name.md"])