"""Two example users for the report, from validation predictions (no tuning, results files unchanged).

Users: the most and the least active cohort-A user in weeks 11-20, activity = number of watched airings
(screen_time >= 0.8); ties -> smaller user_id. Methods: popularity, PPR (v1), MF, CBF, v6, v7.
Missing validation predictions (v6, v7) are rebuilt with the same code that produced results_validation.json
and checked against the stored MAP@5. Marks: hit = in the truth (weeks 21-30), repeat = watched in weeks 11-20,
fallback = slot filled from popularity (the method had fewer than 5 own ids).

Run from the project root:  python -m src.two_users
"""
import json

import numpy as np
import pandas as pd

from src import config, ppr
from src import protocol as P
from src.cbf import item_matrix
from src.cbf import rank_users as cbf_rank_users
from src.hybrid import load_preds
from src.metrics import apk
from src.rerank import FEATURES, add_labels, build_window, fit, rank_by
from src.slots import load_views

METHODS = ["popularity", "ppr", "mf", "cbf", "v6", "v7"]
TITLES = {"popularity": "popularity", "ppr": "PPR (v1, α 0.3)", "mf": "MF (ALS)", "cbf": "CBF",
          "v6": "v6 (PPR v5 + слоты, λ 2)", "v7": "v7 (LGBMRanker поверх v6)"}
OUT_FILE = config.OUTPUTS_DIR / "two_users.md"

log = config.get_logger("18_two_users")


def rebuild_v6_v7(results, av, items_df, truth, cohorts):
    """Validation predictions of v6 and v7 (OOF) for cohort C, plus own-list lengths for the fallback mark."""
    alpha = results["methods"]["ppr"]["best_alpha"]
    v5 = results["improve_signal"]["v2_signal"]["final_v5"]
    cfg = {k: v5[k] for k in ("beta", "half_life", "mode", "t")}
    s = results["improve_slots"]["v5_plus_slots"]
    lam = float(s["table"][s["best"]]["lambda"])
    V, ids, _ = item_matrix(items_df)
    users = cohorts["C"]
    win = build_window(av, items_df, "val", users, alpha, cfg, lam, V, ids)
    df, run = add_labels(win["df"], truth), win["run"]
    perm = np.random.default_rng(config.SEED).permutation(np.array(sorted(users)))
    folds = [set(perm[: len(perm) // 2]), set(perm[len(perm) // 2:])]
    v7 = {}
    for k in range(2):
        model = fit(df[df["user_id"].isin(folds[k])])
        test_df = df[df["user_id"].isin(folds[1 - k])]
        v7.update(rank_by(test_df, model.predict(test_df[FEATURES]), sorted(folds[1 - k]), run))
    own = {"v6": df[df["v6_rank"].notna()].groupby("user_id").size().to_dict(),
           "v7": df.groupby("user_id").size().to_dict()}
    return win["v6_preds"], v7, own


def main():
    results = json.loads(P.RESULTS_FILE.read_text(encoding=config.ENCODING))
    av, items_df = load_views(), P.load_items()
    truth = P.build_truth(av, "val")
    cohorts = P.build_cohorts(truth, P.submission_users())
    train, pool = P.train_views(av, "val"), P.candidate_pool(items_df, "val")

    preds = {m: load_preds(config.INTERIM_DIR / f"preds_val_{m}.parquet") for m in ["popularity", "ppr", "mf", "cbf"]}
    v6, v7, own_len = rebuild_v6_v7(results, av, items_df, truth, cohorts)
    stored = {"v6": results["improve_slots"]["v5_plus_slots"]["table"]["lambda=2.0"]["map@5_A"],
              "v7": results["rerank"]["lgbm_v6"]["oof"]["map@5_A"]}
    for name, pr in (("v6", v6), ("v7", v7)):
        got = P.evaluate(pr, truth, cohorts)["map@5_A"]
        assert got == stored[name], f"{name}: rebuilt MAP@5 A {got} != stored {stored[name]}"
        P.save_preds(pr, config.INTERIM_DIR / f"preds_val_{name}.parquet")
        log.info("%s rebuilt, MAP@5 A %.6f = stored", name, got)
    preds["v6"], preds["v7"] = v6, v7
    for name, key in (("popularity", "popularity"), ("ppr", "ppr"), ("mf", "mf"), ("cbf", "cbf")):
        assert P.evaluate(preds[name], truth, cohorts)["map@5_A"] == results["methods"][key]["map@5_A"]

    # users
    watched = train[train["watched"]]
    activity = watched.groupby("user_id").size().reindex(cohorts["A"], fill_value=0)
    act = activity.rename("n").reset_index()
    heavy = act.sort_values(["n", "user_id"], ascending=[False, True]).iloc[0]
    light = act.sort_values(["n", "user_id"], ascending=[True, True]).iloc[0]
    log.info("heavy %s (%d), light %s (%d)", heavy["user_id"], heavy["n"], light["user_id"], light["n"])

    # own-list lengths for the fallback mark
    graph = ppr.fit_graph(train)
    V, ids, _ = item_matrix(items_df)
    users2 = [heavy["user_id"], light["user_id"]]
    ppr_own, _ = ppr.rank_users(graph, pool, users2, results["methods"]["ppr"]["best_alpha"])
    cbf_own, _ = cbf_rank_users(V, ids, train, float(results["improve"]["ppr"]["best"]["half_life"]),
                                P.SPLITS["val"]["train_weeks"][1], pool, users2)
    mf_users = set(watched["user_id"])
    for u in users2:  # recomputed own lists must agree with the saved predictions
        assert preds["ppr"][u][:len(ppr_own.get(u, [])[:5])] == ppr_own.get(u, [])[:5]
        assert preds["cbf"][u][:len(cbf_own.get(u, [])[:5])] == cbf_own.get(u, [])[:5]

    def own_count(method, u):
        return {"popularity": None, "ppr": len(ppr_own.get(u, [])), "mf": 5 if u in mf_users else 0,
                "cbf": len(cbf_own.get(u, [])), "v6": own_len["v6"].get(u, 0), "v7": own_len["v7"].get(u, 0)}[method]

    by_id = items_df.set_index("tv_show_id")
    counts = P.watched_counts(train)
    target_counts = P.watched_counts(P.in_weeks(av, P.SPLITS["val"]["target_weeks"]))
    seen = set(zip(watched["user_id"], watched["tv_show_id"]))

    def name(i):
        return f"{by_id.at[i, 'tv_show_title']}", f"{by_id.at[i, 'tv_show_category']}"

    lines = ["# Два пользователя: примеры рекомендаций (валидация)\n",
             "Окно истории — недели 11–20, истина — недели 21–30 (top-5 по числу watched-эфиров, порог 0.8). "
             "Прогнозы — валидационные (`outputs/interim/preds_val_*.parquet`; v6 и v7 восстановлены тем же кодом, "
             f"их MAP@5 A совпал с results: v6 {stored['v6']}, v7 {stored['v7']} (OOF)). "
             "Пометки: **попал** — в истине; **повтор** — смотрел в неделях 11–20 (порог 0.8); "
             "**fallback** — место добито popularity.\n",
             f"Выбор: среди {len(cohorts['A'])} пользователей когорты A — самый активный и самый малоактивный по числу "
             "watched-эфиров в неделях 11–20 (при равенстве — меньший user_id).\n"]
    summary = {}
    for label, row in (("Самый активный", heavy), ("Самый малоактивный", light)):
        u = row["user_id"]
        hours = av[(av["user_id"] == u) & av["week"].between(11, 20)]["overlap_s"].sum() / 3600
        lines.append(f"## {label}: `{u}`\n")
        lines.append(f"Watched-эфиров в неделях 11–20: **{row['n']}**; передач в истории: "
                     f"{int((counts['user_id'] == u).sum())}; часов в эфирах с id (все пары): {hours:.1f}.\n")
        hist = P.rank_per_user(counts[counts["user_id"] == u], 10)
        lines.append("**История, top-10 (недели 11–20)**\n")
        if hist.empty:
            lines.append("_Нет ни одного watched-эфира — пользователь cold start; все методы отдают popularity._\n")
        else:
            lines += ["| # | id | Название | Категория | watched-эфиров |", "|---|---|---|---|---|"]
            lines += [f"| {r.rank} | {r.tv_show_id} | {name(r.tv_show_id)[0]} | {name(r.tv_show_id)[1]} | {r.n_watched} |"
                      for r in hist.itertuples()]
            lines.append("")
        tc = target_counts[target_counts["user_id"] == u].set_index("tv_show_id")["n_watched"]
        lines += ["**Истина, недели 21–30**\n", "| # | id | Название | Категория | watched-эфиров | смотрел в 11–20 |",
                  "|---|---|---|---|---|---|"]
        lines += [f"| {k + 1} | {i} | {name(i)[0]} | {name(i)[1]} | {tc[i]} | {'да' if (u, i) in seen else 'нет'} |"
                  for k, i in enumerate(truth[u])]
        lines.append("")
        summary[u] = {"label": label, "watched_airings_11_20": int(row["n"])}
        for m in METHODS:
            ids5 = preds[m][u]
            n_own = own_count(m, u)
            ap = apk(truth[u], ids5)
            summary[u][m] = round(ap, 4)
            lines += [f"**{TITLES[m]}** — AP@5 = {ap:.4f}\n", "| # | id | Название | Категория | попал | повтор | fallback |",
                      "|---|---|---|---|---|---|---|"]
            for k, i in enumerate(ids5):
                fb = "—" if n_own is None else ("да" if k >= n_own else "нет")
                lines.append(f"| {k + 1} | {i} | {name(i)[0]} | {name(i)[1]} | {'да' if i in truth[u] else 'нет'} | "
                             f"{'да' if (u, i) in seen else 'нет'} | {fb} |")
            lines.append("")
    lines += ["## Сводка AP@5\n", "| Пользователь | " + " | ".join(TITLES[m] for m in METHODS) + " |",
              "|---|" + "---|" * len(METHODS)]
    lines += [f"| {v['label']} ({v['watched_airings_11_20']} эфиров) | " + " | ".join(f"{v[m]:.4f}" for m in METHODS) + " |"
              for v in summary.values()]
    OUT_FILE.write_text("\n".join(lines) + "\n", encoding=config.ENCODING)
    log.info("saved %s; AP@5 %s", OUT_FILE.name, summary)


if __name__ == "__main__":
    main()
