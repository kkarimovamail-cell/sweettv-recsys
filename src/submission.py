"""Write and validate Kaggle submissions in the exact layout of the template.

Run from the project root:  python -m src.submission   (popularity trained on weeks 11-30)
"""
from pathlib import Path

from src import config
from src import protocol as P
from src.baselines import popularity_preds

log = config.get_logger("06_submission")


def read_lines(path: Path) -> list[str]:
    return path.read_text(encoding=config.ENCODING).splitlines()


def write_submission(preds: dict[str, list[str]], path: Path, template: Path = config.SUBMISSION_TEMPLATE) -> None:
    """Rows in the template's user order; LF line endings and no trailing newline, like the template."""
    header, *rows = read_lines(template)
    users = [r.split(",")[0] for r in rows]
    lines = [header] + [f"{u},{' '.join(preds[u])}" for u in users]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding=config.ENCODING)


def validate_submission(path: Path, pool: set[str], template: Path = config.SUBMISSION_TEMPLATE) -> list[str]:
    """Return a list of problems (empty = valid)."""
    t_header, *t_rows = read_lines(template)
    header, *rows = read_lines(path)
    errors = []
    if header != t_header:
        errors.append(f"header {header!r} != template {t_header!r}")
    t_users = [r.split(",")[0] for r in t_rows]
    users = [r.split(",")[0] for r in rows]
    if len(users) != len(t_users):
        errors.append(f"{len(users)} rows != template {len(t_users)}")
    elif users != t_users:
        errors.append(f"user_id order differs from template at row {next(i for i, (a, b) in enumerate(zip(users, t_users)) if a != b) + 1}")
    for n, row in enumerate(rows, start=1):
        parts = row.split(",")
        if len(parts) != 2:
            errors.append(f"row {n}: {len(parts)} columns")
            continue
        ids = parts[1].split(" ")
        if len(ids) != config.K:
            errors.append(f"row {n}: {len(ids)} ids")
        if len(set(ids)) != len(ids):
            errors.append(f"row {n}: repeated ids")
        if any(not i.isdigit() or int(i) == 0 for i in ids):
            errors.append(f"row {n}: zero or non-numeric id")
        elif any(i != str(int(i)) for i in ids):
            errors.append(f"row {n}: non-canonical id (leading zeros)")
        outside = [i for i in ids if i not in pool]
        if outside:
            errors.append(f"row {n}: ids outside the pool {outside}")
    return errors


def main():
    av, items = P.load_airing_views(), P.load_items()
    users = P.submission_users()
    train, pool = P.train_views(av, "test"), P.candidate_pool(items, "test")
    preds, fill_stats = popularity_preds(train, pool, users)
    path = config.SUBMISSIONS_DIR / "submission_popularity.csv"
    write_submission(preds, path)
    errors = validate_submission(path, pool)
    log.info("%s: train weeks %s, pool %d, fill %s", path.name, P.SPLITS["test"]["train_weeks"], len(pool), fill_stats)
    log.info("validation: %s", "OK" if not errors else errors[:10])
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
