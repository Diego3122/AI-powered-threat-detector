from scripts import repo_safety_check


def test_repo_safety_check_passes_for_current_repo():
    findings = repo_safety_check.run_checks()

    assert findings == []


def test_placeholder_detection_accepts_expected_example_values():
    assert repo_safety_check._looks_like_placeholder("replace-with-strong-db-password")
    assert repo_safety_check._looks_like_placeholder("change-me")
    assert repo_safety_check._looks_like_placeholder("")


def test_placeholder_detection_rejects_realistic_secret_values():
    assert not repo_safety_check._looks_like_placeholder("a9D2mK4pQ7xT8wZ1nB6cR3vL0sH5yF8u")


def test_sensitive_file_staged_for_deletion_is_not_reported(monkeypatch):
    responses = {
        ("ls-files",): {".env.production"},
        ("diff", "--cached", "--name-only", "--diff-filter=ACMR"): set(),
        ("diff", "--cached", "--name-only", "--diff-filter=D"): {".env.production"},
    }

    monkeypatch.setattr(
        repo_safety_check,
        "_git_list_paths",
        lambda *args: responses.get(args, set()),
    )

    assert repo_safety_check._check_git_tracking_of_sensitive_files() == []


def test_iter_repo_files_skips_private_env_files(monkeypatch):
    class FakePath:
        def __init__(self, name: str):
            self.name = name
            self.suffix = ""
            if "." in name and not name.startswith("."):
                self.suffix = f".{name.rsplit('.', 1)[1]}"
            self.parts = ("repo", name)

        def is_file(self) -> bool:
            return True

    class FakeRoot:
        def rglob(self, _pattern: str):
            return [
                FakePath(".env.example"),
                FakePath(".env.local"),
                FakePath("notes.md"),
            ]

    monkeypatch.setattr(repo_safety_check, "REPO_ROOT", FakeRoot())

    files = repo_safety_check._iter_repo_files()

    assert [path.name for path in files] == ["notes.md"]
