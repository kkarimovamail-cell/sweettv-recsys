"""Comparison of every validation variant, recomputed from the saved prediction files.

  MAP@5 A / C          protocol metric (checked against results_validation.json)
  hit >= 1 A / C       share of users with at least one of the 5 ids in their truth
  repeats A            share of recommended slots the user already watched in the train window (weeks 11-20)
  mean popularity A    mean number of distinct train-window users with a watched airing of the recommended programme
  distinct ids A       number of distinct programmes recommended across cohort A
  runtime              see RUNTIME_NOTES

Run from the project root:  python -m src.comparison   (after baselines, ppr, mf, hybrid)
"""
import json

import numpy as np
import pandas as pd

from src import config
from src import protocol as P
from src.hybrid import load_preds

RUNTIME_NOTES = {
    "baseline": "ranking + fill for 4,030 users",
    "ppr": "PageRank iterations + ranking for 4,030 users; graph build is shared and listed once",
    "mf": "ALS fit + scoring for 4,030 users",
    "hybrid": "PPR graph build + ranking + MF fit/ranking (depth 50) + fusion",
}
COLUMNS = ["family", "variant", "map@5_A", "map@5_C", "hit_A", "hit_C", "repeats_A",
           "mean_popularity_A", "distinct_ids_A", "runtime_s", "selected"]


def variants(results: dict) -> list[dict]:
    m = results["methods"]
    out = [{"family": "baseline", "variant": name, "file": f"preds_val_{name}.parquet",
            "stored": m[name], "runtime_s": m[name]["runtime_s"], "selected": True}
           for name in ("popularity", "personal_history")]
    for a, v in m["ppr"]["grid"].items():
        out.append({"family": "ppr", "variant": f"alpha={a}", "file": f"preds_val_ppr_a{float(a):g}.parquet",
                    "stored": v, "runtime_s": v["runtime_s"], "selected": float(a) == m["ppr"]["best_alpha"]})
    for k, v in m["mf"]["grid"].items():
        out.append({"family": "mf", "variant": k, "file": f"preds_val_mf_{k}.parquet",
                    "stored": v, "runtime_s": v["runtime_s"], "selected": k == m["mf"]["best"]})
    for k, v in m["hybrid_rrf"]["grid"].items():
        out.append({"family": "hybrid", "variant": f"w={float(k):g}", "file": f"preds_val_hybrid_w{float(k):g}.parquet",
                    "stored": v, "runtime_s": v["runtime_s"], "selected": float(k) == m["hybrid_rrf"]["best_w"]})
    for k, v in m["mf"]["off_grid"]["variants"].items():
        out.append({"family": "mf_off_grid", "variant": k, "file": f"preds_val_mf_{k}.parquet",
                    "stored": v, "runtime_s": v["runtime_s"], "selected": False})
    return out


def row_metrics(preds: dict, truth: dict, cohorts: dict, train: pd.DataFrame, popularity: pd.Series) -> dict:
    A, C = cohorts["A"], cohorts["C"]
    hit = {name: float(np.mean([bool(set(preds[u]) & set(truth[u])) for u in users])) for name, users in cohorts.items()}
    slots = [i for u in A for i in preds[u]]
    return {**P.evaluate(preds, truth, cohorts), "hit_A": round(hit["A"], 4), "hit_C": round(hit["C"], 4),
            "repeats_A": P.history_share(preds, train, A),
            "mean_popularity_A": round(float(popularity.reindex(slots, fill_value=0).mean()), 1),
            "distinct_ids_A": len(set(slots))}


def to_markdown(df: pd.DataFrame) -> str:
    head = "| " + " | ".join(df.columns) + " |\n|" + "---|" * len(df.columns) + "\n"
    return head + "".join("| " + " | ".join(str(v) for v in r) + " |\n" for r in df.itertuples(index=False))


def main():
    results = json.loads(P.RESULTS_FILE.read_text(encoding=config.ENCODING))
    av, items = P.load_airing_views(), P.load_items()
    truth = P.build_truth(av, "val")
    cohorts = P.build_cohorts(truth, P.submission_users())
    train = P.train_views(av, "val")
    popularity = train[train["watched"]].groupby("tv_show_id")["user_id"].nunique()
    n_train_users = int(train.loc[train["watched"], "user_id"].nunique())

    rows = []
    for v in variants(results):
        preds = load_preds(config.INTERIM_DIR / v["file"])
        met = row_metrics(preds, truth, cohorts, train, popularity)
        for key in ("map@5_A", "map@5_C"):
            assert abs(met[key] - v["stored"][key]) < 1e-9, f"{v['variant']}: {key} differs from results"
        rows.append({"family": v["family"], "variant": v["variant"], **met,
                     "runtime_s": v["runtime_s"], "selected": "yes" if v["selected"] else ""})
    df = pd.DataFrame(rows)[COLUMNS]
    main_df, off = df[df["family"] != "mf_off_grid"], df[df["family"] == "mf_off_grid"]

    graph_s = results["methods"]["ppr"]["graph"]["build_s"]
    md = [
        "# Method comparison (validation: train weeks 11–20, truth weeks 21–30)\n",
        f"Cohort A = {len(cohorts['A'])} submission users, cohort C = {len(cohorts['C'])} log users "
        "(each with ≥ 5 programmes in the truth). All variants are listed, including the worst. "
        "`selected` marks the variant each family selected on MAP@5 A.\n",
        "Columns: **hit_A / hit_C** = share of users with ≥ 1 of the 5 ids in their truth; "
        "**repeats_A** = share of recommended slots already watched by the user in weeks 11–20; "
        f"**mean_popularity_A** = mean number of distinct users (of {n_train_users} with a watched airing in weeks 11–20) "
        "who watched the recommended programme in weeks 11–20; **distinct_ids_A** = distinct programmes "
        "recommended across cohort A; **runtime_s** — " + "; ".join(f"{k}: {v}" for k, v in RUNTIME_NOTES.items())
        + f" (PPR graph build: {graph_s} s).\n",
        "## All methods\n", to_markdown(main_df),
        "\n## MF diagnostics — off-grid, not used for selection (вне сетки, для выбора не использовались)\n",
        to_markdown(off),
        "\n## Checks\n",
        f"- Hybrid: {json.dumps(results['methods']['hybrid_rrf']['checks'], ensure_ascii=False)}\n",
        f"- MF reproducibility (best config fitted twice): "
        f"{json.dumps({k: v for k, v in results['methods']['mf']['reproducibility'].items() if k != 'map_again'})}\n",
        "- MAP@5 recomputed here from the saved prediction files equals the stored value for every row.\n",
    ]
    out = config.OUTPUTS_DIR / "comparison.md"
    out.write_text("\n".join(md), encoding=config.ENCODING)
    P.update_results("comparison", "table", main_df.to_dict(orient="records"))
    P.update_results("comparison", "off_grid_not_used_for_selection", off.to_dict(orient="records"))
    P.update_results("comparison", "columns", {c: d for c, d in [
        ("hit", "share of users with >= 1 hit"), ("repeats_A", "share of slots watched by the user in train"),
        ("mean_popularity_A", f"mean distinct train users per recommended programme (of {n_train_users})"),
        ("runtime_s", RUNTIME_NOTES)]})
    print(to_markdown(main_df))
    print(to_markdown(off))


if __name__ == "__main__":
    main()
