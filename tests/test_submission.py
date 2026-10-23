import pytest

from src.submission import validate_submission, write_submission

POOL = {"1", "2", "3", "4", "5", "6", "7", "007"}


@pytest.fixture
def template(tmp_path):
    path = tmp_path / "template.csv"
    path.write_text("user_id,tv_show_id\n111,0 0 0 0 0\n222,0 0 0 0 0", encoding="utf-8")
    return path


def write(tmp_path, text):
    path = tmp_path / "sub.csv"
    path.write_text(text, encoding="utf-8")
    return path


def test_written_file_is_valid_and_keeps_template_order(tmp_path, template):
    path = tmp_path / "sub.csv"
    write_submission({"222": ["6", "5", "4", "3", "2"], "111": ["1", "2", "3", "4", "7"]}, path, template)
    assert path.read_text(encoding="utf-8") == "user_id,tv_show_id\n111,1 2 3 4 7\n222,6 5 4 3 2"
    assert validate_submission(path, POOL, template) == []


@pytest.mark.parametrize("text, problem", [
    ("user,tv_show_id\n111,1 2 3 4 5\n222,1 2 3 4 5", "header"),
    ("user_id,tv_show_id\n222,1 2 3 4 5\n111,1 2 3 4 5", "order"),
    ("user_id,tv_show_id\n111,1 2 3 4 5", "rows"),
    ("user_id,tv_show_id\n111,1 2 3 4\n222,1 2 3 4 5", "4 ids"),
    ("user_id,tv_show_id\n111,1 1 3 4 5\n222,1 2 3 4 5", "repeated"),
    ("user_id,tv_show_id\n111,0 2 3 4 5\n222,1 2 3 4 5", "zero"),
    ("user_id,tv_show_id\n111,1 2 3 4 99\n222,1 2 3 4 5", "outside the pool"),
    ("user_id,tv_show_id\n111,1 2 3 4 007\n222,1 2 3 4 5", "non-canonical"),
])
def test_invalid_files_are_caught(tmp_path, template, text, problem):
    errors = validate_submission(write(tmp_path, text), POOL, template)
    assert any(problem in e for e in errors), errors
