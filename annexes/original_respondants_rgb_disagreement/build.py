#!/usr/bin/env python3
"""
Respondent agreement on identical RGB values, at two levels of detail.

Two people are occasionally shown the exact same pixel. How often do they give it
the same name? Restricted to pixels seen by exactly two people (one repetition):
that is ~95% of all repeated pixels, and it keeps every number on one clean
population instead of mixing populations of different sizes.

Dataset level: one agreement / chance / kappa row per colour count.
Per-class level: the same pixels, broken down by name. A disagreement involves two
names at once, so instead of attributing it to one side, condition on each in turn:
for name c, take every pixel where someone said c and ask what the other person
said. A disagreeing pair (A,B) therefore contributes once to A's row (as B) and
once to B's row (as A) — symmetric, nothing double counted within a row, and
agreement is simply the diagonal.

    python annexes/original_respondants_rgb_disagreement/build.py            # all four
    python annexes/original_respondants_rgb_disagreement/build.py 96         # just one

Writes README.md, agreement_<n>c.png, agreement_per_class_<n>c.csv and
contested_pairs_<n>c.csv into this directory.
"""
import argparse
import os
from collections import Counter, defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
DEFAULT_CSV = os.path.join(ROOT, "mainsurvey.csv")

N_PARTNERS = 3
MAX_BARS = 60          # above this, chart only the most- and least-agreeing extremes
REPEAT_COUNTS = (2, 3, 4, 5)


def paired_answers(df, top_n):
    """The shared primitive: for the top-n names, every pixel seen by exactly two
    people, as their two answers — plus the repeat-count histogram and each name's
    mean RGB, which both the dataset- and class-level views are built from."""
    vc = df["colorname"].value_counts()
    d = df[df["colorname"].isin(vc.nlargest(top_n).index)].copy()
    d["key"] = d.r.values * 65536 + d.g.values * 256 + d.b.values

    sizes = d["key"].value_counts()
    repeats = {n: int((sizes == n).sum()) for n in REPEAT_COUNTS}
    two = sizes[sizes == 2].index
    sub = d[d["key"].isin(two)].sort_values("key")
    first = sub.groupby("key")["colorname"].nth(0).values
    second = sub.groupby("key")["colorname"].nth(1).values

    rgb = d.groupby("colorname")[["r", "g", "b"]].mean() / 255.0
    return {"first": first, "second": second, "distinct_rgb": len(sizes),
            "repeats": repeats, "rgb": rgb}


def agreement_summary(paired):
    """One dataset-level row: agreement, chance agreement, and Cohen's kappa.

    chance = what two people would score by ignoring the pixel and answering from
    name popularity alone, i.e. sum(p_i^2) over the pooled pair of answers.
    kappa  = agreement with chance matches removed: (agreement - chance) / (1 - chance).
    """
    first, second = paired["first"], paired["second"]
    n = len(first)
    agreement = float((first == second).mean())
    marginal = pd.Series(np.concatenate([first, second])).value_counts(normalize=True)
    chance = float((marginal ** 2).sum())
    kappa = (agreement - chance) / (1 - chance)
    return {"n_pairs": n, "distinct_rgb": paired["distinct_rgb"],
            "pct_of_distinct": n / paired["distinct_rgb"], "disagree": 1 - agreement,
            "agreement": agreement, "chance": chance, "kappa": kappa}


def per_class_agreement(paired):
    """Per-name agreement table + the ranked list of confusable name pairs."""
    agree, involved = Counter(), Counter()
    partners = defaultdict(Counter)
    for a, b in zip(paired["first"], paired["second"]):
        if a == b:
            agree[a] += 1
            involved[a] += 1
        else:
            partners[a][b] += 1
            partners[b][a] += 1
            involved[a] += 1
            involved[b] += 1

    rows = []
    for c in involved:
        top = partners[c].most_common(N_PARTNERS)
        row = {"colorname": c, "pixels_with_this_name": involved[c],
               "both_agreed": agree[c], "agreement": agree[c] / involved[c]}
        for i in range(N_PARTNERS):
            name, count = top[i] if i < len(top) else ("", 0)
            row[f"partner_{i+1}_name"] = name
            row[f"partner_{i+1}_count"] = count
        rows.append(row)
    table = pd.DataFrame(rows).sort_values("agreement", ascending=False).reset_index(drop=True)

    pairs, seen = [], set()
    for c in partners:
        for p, n in partners[c].items():
            k = tuple(sorted((c, p)))
            if k not in seen:
                seen.add(k)
                pairs.append({"name_a": k[0], "name_b": k[1], "disagreements": n})
    pairs = pd.DataFrame(pairs).sort_values("disagreements", ascending=False).reset_index(drop=True)
    return table, pairs


def chart(table, rgb, top_n):
    """One stacked bar per name: agreement, then its top confusion partners, then the rest."""
    ordered = table.sort_values("agreement")
    subset = len(ordered) > MAX_BARS
    if subset:
        half = MAX_BARS // 2
        ordered = pd.concat([ordered.head(half), ordered.tail(half)])
        note = (f"showing only the {half} least- and {half} most-agreeing names "
                f"of {len(table)} (full list in the CSV)")
    else:
        note = f"all {len(ordered)} names"

    fig, ax = plt.subplots(figsize=(11, max(4, 0.19 * len(ordered))))
    for i, row in enumerate(ordered.itertuples()):
        tot = row.pixels_with_this_name
        agreed = row.both_agreed / tot
        left = agreed
        ax.barh(i, left, color=rgb.loc[row.colorname].values, edgecolor="black", linewidth=.4)
        for j in range(1, N_PARTNERS + 1):
            p = getattr(row, f"partner_{j}_name")
            if not p:
                continue
            w = getattr(row, f"partner_{j}_count") / tot
            ax.barh(i, w, left=left, color=rgb.loc[p].values,
                    edgecolor="white", linewidth=.4, alpha=.9)
            if w > .06:
                ax.text(left + w / 2, i, p, ha="center", va="center", fontsize=5.5,
                        color="white" if rgb.loc[p].values.mean() < .55 else "black")
            left += w
        ax.barh(i, 1 - left, left=left, color="#dddddd", edgecolor="white", linewidth=.4)
        ax.text(1.015, i, f"{agreed:.0%}", ha="left", va="center",
                fontsize=6, fontweight="bold", color="#333333")
        if subset and i == MAX_BARS // 2 - 1:
            ax.axhline(i + .5, color="black", ls=":", lw=1)

    ax.set_yticks(range(len(ordered)))
    ax.set_yticklabels(ordered["colorname"], fontsize=6)
    ax.set_xlim(0, 1.07)
    ax.set_ylim(-.6, len(ordered) - .4)
    ax.set_xticks([0, .2, .4, .6, .8, 1])
    ax.set_xlabel("share of pixels where this name was said (left block = both people agreed)")
    ax.set_title(f"Who each colour name is confused with — {top_n} colours, pixels seen exactly twice\n"
                 f"left block = agreement; segments coloured by the other person's answer; "
                 f"grey = all remaining names\n{note}",
                 fontsize=10, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    p = os.path.join(HERE, f"agreement_{top_n}c.png")
    plt.savefig(p, dpi=160, bbox_inches="tight")
    plt.close()
    return p


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("counts", nargs="*", type=int, default=[14, 96, 797, 5363])
    ap.add_argument("--csv", default=DEFAULT_CSV)
    args = ap.parse_args()

    df = pd.read_csv(args.csv, usecols=["r", "g", "b", "colorname"])

    dataset_rows, repeat_rows, blocks = [], [], []
    for n in args.counts:
        paired = paired_answers(df, n)
        s = agreement_summary(paired)
        table, pairs = per_class_agreement(paired)
        table.to_csv(os.path.join(HERE, f"agreement_per_class_{n}c.csv"), index=False)
        pairs.to_csv(os.path.join(HERE, f"contested_pairs_{n}c.csv"), index=False)
        png = chart(table, paired["rgb"], n)

        dataset_rows.append({"labels": n, **s})
        repeat_rows.append({"labels": n, **paired["repeats"]})
        print(f"{n:>5} colours -> {len(table)} names, agreement {s['agreement']:.3f}, "
              f"kappa {s['kappa']:.3f}")

        b = [f"\n### {n} colours\n", f"![agreement]({os.path.basename(png)})\n",
             "**Most agreed**\n",
             "| name | pixels | agreement | mostly confused with |", "|---|---|---|---|"]
        for _, r in table.head(8).iterrows():
            partner = f"{r.partner_1_name} ({r.partner_1_count})" if r.partner_1_name else ""
            b.append(f"| {r.colorname} | {r.pixels_with_this_name:,} | {r.agreement:.0%} | {partner} |")
        b += ["\n**Least agreed**\n",
              "| name | pixels | agreement | mostly confused with |", "|---|---|---|---|"]
        for _, r in table.tail(8).iloc[::-1].iterrows():
            partner = f"{r.partner_1_name} ({r.partner_1_count})" if r.partner_1_name else ""
            b.append(f"| {r.colorname} | {r.pixels_with_this_name:,} | {r.agreement:.0%} | {partner} |")
        b += ["\n**Most contested pairs**\n", "| name A | name B | disagreements |", "|---|---|---|"]
        for _, r in pairs.head(8).iterrows():
            b.append(f"| {r.name_a} | {r.name_b} | {r.disagreements:,} |")
        blocks.append("\n".join(b))

    write_readme(dataset_rows, repeat_rows, blocks)
    print(f"\nwrote {HERE}/README.md")


def write_readme(dataset_rows, repeat_rows, blocks):
    head = [
        "# Respondent disagreement on identical RGB values\n",
        "How often two people shown the **identical pixel** give it the same name.\n",
        "**Method.** Only pixels answered by exactly two people (one repetition) — ~95% of all",
        "repeated pixels, which keeps every number on one population. A disagreement involves two",
        "names, so instead of attributing it to one side we condition on each in turn: for name *c*,",
        "take every pixel where someone said *c* and ask what the other person said. A pair (A,B)",
        "counts once in A's row (as B) and once in B's row (as A).\n",
        "## Dataset level\n",
        "*agreement* = the two people gave the same name. *chance* = what two people would score",
        "by ignoring the pixel and answering from name popularity alone. *kappa* (Cohen's kappa) is",
        "agreement with those lucky matches removed: (agreement − chance) / (1 − chance);",
        "0 = no better than guessing, 1 = perfect.\n",
        "| labels | distinct RGB | seen exactly 2x | % of all RGB | disagree | agreement | chance | kappa |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in dataset_rows:
        head.append(f"| {r['labels']} | {r['distinct_rgb']:,} | {r['n_pairs']:,} | "
                    f"{r['pct_of_distinct']:.1%} | {r['disagree']:.1%} | {r['agreement']:.3f} | "
                    f"{r['chance']:.3f} | {r['kappa']:.2f} |")

    head += ["\nHow often a pixel repeats at all (share of all distinct RGB values):\n",
             "| labels | " + " | ".join(f"seen {n}x" for n in REPEAT_COUNTS) + " |",
             "|---|" + "---|" * len(REPEAT_COUNTS)]
    for rr, dr in zip(repeat_rows, dataset_rows):
        cells = " | ".join(f"{rr[n]:,} ({rr[n] / dr['distinct_rgb']:.2%})" for n in REPEAT_COUNTS)
        head.append(f"| {rr['labels']} | {cells} |")

    head += [
        "\nPixels seen twice carry ~95% of all repeated pixels, which is why the per-class",
        "analysis below rests on them. Agreement is a *lower bound* on the naming ceiling",
        "(predicting the mode beats matching a random draw).\n",
        "## Per-class level\n",
        "**Reading the charts.** One bar per name = 100% of the pixels where that name was said.",
        "The left block is agreement, coloured with the name's own average RGB; the next segments",
        "are the three most common alternative answers, each in *its* own colour; grey is everything",
        "else. Bars are sorted by agreement, with the percentage on the right.\n",
        "**Files per colour count:** `agreement_<n>c.png` (chart), "
        "`agreement_per_class_<n>c.csv` (every name), `contested_pairs_<n>c.csv` (every pair).\n",
    ]

    with open(os.path.join(HERE, "README.md"), "w") as f:
        f.write("\n".join(head) + "\n" + "\n".join(blocks) + "\n")


if __name__ == "__main__":
    main()
