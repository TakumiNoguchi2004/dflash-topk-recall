"""Recall@k analysis for DFlash's drafter-top-k-vs-verifier-token experiment
(xPress side-study; see run_dflash_topk_recall.py for how the raw ranks are
collected).

For every block-diffusion draft round, run_dflash_topk_recall.py records the
rank of the target's per-position judgment inside the drafter's own logit
ordering, at every position in the block (rank 0 = drafter's top-1). Using
each round's accept_lengths entry (k = accepted candidates before the
reject/bonus position), this script splits those ranks into:

  - pre-reject  (position index < k): trivially rank 0 always (accept means
    top-1 matched) -- included only as a sanity check, not the interesting
    number.
  - post-reject (position index >= k): the reject position itself plus
    everything drafted after it in the same block. This is where xPress's
    claim lives: rejection only means the drafter's top-1 guess was wrong,
    not that its whole predicted distribution was bad -- the question is
    just whether the target's actual token sits inside the drafter's top-k
    at that position.

recall@k = fraction of ranks < k (rank is 0-indexed, so k=1 means "top-1
matched"). Only the reject slot itself is guaranteed rank > 0 (that's what
"reject" means); positions after it in the block were never actually
checked by the real accept/reject walk, so their draft top-1 can coincide
with the target's per-position judgment by chance -- recall@1 on the
post-reject subset is NOT trivially 0, and its value is itself informative
(how often the "wasted" remainder of a rejected block would have kept going
under naive top-1 sampling, before any recovery mechanism).

If recall@k rises sharply above recall@1 for small k, that's xPress's claim
confirmed: the target's token is usually a near-miss, not a wild one. Two
natural ways to cash that in: rerank the drafter's own top-k to pick a
single better candidate per position (still one path, verified as usual),
or keep multiple candidates explicit as branches of a draft tree (like
EAGLE's dynamic tree) and verify several paths in parallel instead of
committing to one guess.

Emits a CSV and a self-contained HTML report (open via VSCode preview, not
published) with the recall@k curve.

Usage:
    python analyze_dflash_topk_recall.py --run-name v1
"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"

K_VALUES = [1, 2, 4, 8, 16, 32, 64]


def read_jsonl(path: Path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def split_ranks(record: dict) -> tuple[list[int], list[int]]:
    """Returns (pre_reject_ranks, post_reject_ranks) for one question's record."""
    gamma = record["gamma"]
    ranks = record["token_ranks"]
    pre, post = [], []
    for i, k in enumerate(record["accept_lengths"]):
        round_ranks = ranks[i * gamma : (i + 1) * gamma]
        pre.extend(round_ranks[:k])
        post.extend(round_ranks[k:])
    return pre, post


def recall_at_k(ranks: list[int], k_values: list[int]) -> dict[int, float]:
    n = len(ranks)
    if n == 0:
        return {k: float("nan") for k in k_values}
    return {k: sum(1 for r in ranks if r < k) / n for k in k_values}


def write_csv(path: Path, pre_recall, post_recall, n_pre, n_post):
    with open(path, "w", encoding="utf-8") as f:
        f.write("k,recall_pre_reject,recall_post_reject\n")
        for k in K_VALUES:
            f.write(f"{k},{pre_recall[k]:.4f},{post_recall[k]:.4f}\n")
    print(f"wrote {path} (n_pre_reject={n_pre}, n_post_reject={n_post})")


def build_svg_chart(post_recall, pre_recall, width=760, height=420):
    pad_left, pad_right, pad_top, pad_bottom = 56, 24, 24, 44
    plot_w = width - pad_left - pad_right
    plot_h = height - pad_top - pad_bottom
    x_max = len(K_VALUES) - 1

    def sx(i):
        return pad_left + i / x_max * plot_w if x_max else pad_left

    def sy(y):
        return pad_top + (1 - y) * plot_h

    def path_for(values):
        pts = [(sx(i), sy(values[k])) for i, k in enumerate(K_VALUES)]
        return "M " + " L ".join(f"{x:.1f},{y:.1f}" for x, y in pts)

    gridlines = []
    for yt in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
        y = sy(yt)
        gridlines.append(
            f'<line x1="{pad_left}" y1="{y:.1f}" x2="{width - pad_right}" y2="{y:.1f}" class="gridline" />'
            f'<text x="{pad_left - 10}" y="{y + 4:.1f}" class="tick-label" text-anchor="end">{yt:.1f}</text>'
        )
    x_labels = "".join(
        f'<text x="{sx(i):.1f}" y="{height - pad_bottom + 20}" class="tick-label" text-anchor="middle">{k}</text>'
        for i, k in enumerate(K_VALUES)
    )

    def markers_for(values, cls):
        return "\n".join(
            f'<circle cx="{sx(i):.1f}" cy="{sy(values[k]):.1f}" r="4" class="marker {cls}" '
            f'data-k="{k}" data-value="{values[k]:.4f}" />'
            for i, k in enumerate(K_VALUES)
        )

    svg = f'''
<svg viewBox="0 0 {width} {height}" class="chart-svg" role="img" aria-label="Recall@k of verifier token within drafter top-k">
  <g class="gridlines">{''.join(gridlines)}</g>
  <line x1="{pad_left}" y1="{height - pad_bottom}" x2="{width - pad_right}" y2="{height - pad_bottom}" class="axis-line" />
  <g class="x-ticks">{x_labels}</g>
  <text x="{pad_left + plot_w / 2:.1f}" y="{height - 4}" class="axis-title" text-anchor="middle">k</text>
  <text x="16" y="{pad_top + plot_h / 2:.1f}" class="axis-title" text-anchor="middle" transform="rotate(-90, 16, {pad_top + plot_h / 2:.1f})">recall@k</text>
  <path d="{path_for(post_recall)}" class="line series-1" fill="none" />
  <path d="{path_for(pre_recall)}" class="line series-2" fill="none" />
  <g class="markers">{markers_for(post_recall, 'series-1')}{markers_for(pre_recall, 'series-2')}</g>
</svg>'''
    return svg


HTML_TEMPLATE = """<!doctype html>
<html>
<head>
<meta charset="utf-8" />
<title>DFlash drafter top-k recall -- {run_name}</title>
<style>
  .viz-root {{
    color-scheme: light;
    --surface-1: #fcfcfb;
    --page: #f9f9f7;
    --text-primary: #0b0b0b;
    --text-secondary: #52514e;
    --text-muted: #898781;
    --gridline: #e1e0d9;
    --axis: #c3c2b7;
    --series-1: #2a78d6;
    --series-2: #898781;
  }}
  @media (prefers-color-scheme: dark) {{
    :root:where(:not([data-theme="light"])) .viz-root {{
      color-scheme: dark;
      --surface-1: #1a1a19;
      --page: #0d0d0d;
      --text-primary: #ffffff;
      --text-secondary: #c3c2b7;
      --text-muted: #898781;
      --gridline: #2c2c2a;
      --axis: #383835;
      --series-1: #3987e5;
      --series-2: #6b6a66;
    }}
  }}
  :root[data-theme="dark"] .viz-root {{
    color-scheme: dark;
    --surface-1: #1a1a19;
    --page: #0d0d0d;
    --text-primary: #ffffff;
    --text-secondary: #c3c2b7;
    --text-muted: #898781;
    --gridline: #2c2c2a;
    --axis: #383835;
    --series-1: #3987e5;
    --series-2: #6b6a66;
  }}
  * {{ box-sizing: border-box; }}
  body {{ margin: 0; font-family: system-ui, -apple-system, "Segoe UI", sans-serif; background: var(--page); color: var(--text-primary); }}
  .viz-root {{ max-width: 820px; margin: 0 auto; padding: 32px 20px 60px; }}
  h1 {{ font-size: 1.3rem; margin: 0 0 4px; }}
  .subtitle {{ color: var(--text-secondary); font-size: 0.9rem; margin: 0 0 24px; }}
  .card {{ background: var(--surface-1); border: 1px solid var(--gridline); border-radius: 12px; padding: 20px; margin-bottom: 24px; overflow-x: auto; }}
  .legend {{ display: flex; gap: 20px; margin-bottom: 8px; font-size: 0.85rem; color: var(--text-secondary); }}
  .legend-item {{ display: flex; align-items: center; gap: 6px; }}
  .legend-key {{ width: 16px; height: 2px; border-radius: 1px; }}
  .chart-svg {{ width: 100%; height: auto; overflow: visible; }}
  .gridline {{ stroke: var(--gridline); stroke-width: 1; }}
  .axis-line {{ stroke: var(--axis); stroke-width: 1; }}
  .tick-label {{ fill: var(--text-muted); font-size: 11px; }}
  .axis-title {{ fill: var(--text-secondary); font-size: 12px; }}
  .line {{ stroke-width: 2; stroke-linejoin: round; stroke-linecap: round; }}
  .series-1 {{ stroke: var(--series-1); }}
  .series-2 {{ stroke: var(--series-2); }}
  .marker {{ stroke: var(--surface-1); stroke-width: 2; }}
  .marker.series-1 {{ fill: var(--series-1); }}
  .marker.series-2 {{ fill: var(--series-2); }}
  table {{ border-collapse: collapse; width: 100%; font-size: 0.85rem; }}
  th, td {{ text-align: right; padding: 6px 12px; border-bottom: 1px solid var(--gridline); font-variant-numeric: tabular-nums; }}
  th:first-child, td:first-child {{ text-align: left; }}
  th {{ color: var(--text-secondary); font-weight: 600; }}
  .caveats {{ color: var(--text-secondary); font-size: 0.85rem; line-height: 1.6; }}
  .caveats li {{ margin-bottom: 6px; }}
</style>
</head>
<body>
<div class="viz-root">
  <h1>DFlash: does the drafter's top-k contain the verifier's token?</h1>
  <p class="subtitle">run: {run_name} &middot; task: mtbench &middot; n_questions: {n_questions} &middot; n_post_reject_positions: {n_post} &middot; n_pre_reject_positions: {n_pre}</p>

  <div class="card">
    <div class="legend">
      <div class="legend-item"><span class="legend-key" style="background:var(--series-1)"></span>post-reject positions (reject slot + rest of block)</div>
      <div class="legend-item"><span class="legend-key" style="background:var(--series-2)"></span>pre-reject positions (sanity check -- trivially 1.0 for k&ge;1)</div>
    </div>
    {svg}
    <table>
      <thead><tr><th>k</th><th>recall@k (post-reject)</th><th>recall@k (pre-reject)</th></tr></thead>
      <tbody>{rows}</tbody>
    </table>
  </div>

  <div class="card">
    <strong>Reading notes</strong>
    <ul class="caveats">
      <li>Only the reject slot itself is guaranteed top-1 mismatch (rank &gt; 0); positions after it in the block were never actually checked by the real accept/reject walk, so recall@1 on post-reject is NOT trivially 0 -- it's the fraction of "wasted" post-reject positions where the draft's top-1 happened to coincide with the target's judgment anyway.</li>
      <li>Rejection means the drafter's top-1 guess was wrong, not that its whole predicted distribution was bad -- the question this measures is just whether the target's actual token sits inside the drafter's top-k.</li>
      <li>A steep rise from recall@1 toward recall@64 is xPress's claim confirmed for DFlash: most rejections are near-misses against a ~128k-token vocabulary, not wild mispredictions. Turning that into a real speedup needs a way to surface the right candidate: rerank the drafter's own top-k into a single better single-path guess, or keep several top-k candidates explicit as branches of a draft tree (like EAGLE's dynamic tree) and verify multiple paths at once instead of committing to one.</li>
      <li>This measures containment only, not whether a real reranking scheme could actually identify the right candidate within top-k without an oracle -- that's a separate, harder question.</li>
    </ul>
  </div>
</div>
</body>
</html>
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", required=True)
    args = parser.parse_args()

    run_dir = RESULTS_DIR / args.run_name
    in_path = run_dir / "dflash_topk_recall_mtbench.jsonl"

    pre_ranks, post_ranks = [], []
    n_questions = 0
    for record in read_jsonl(in_path):
        pre, post = split_ranks(record)
        pre_ranks.extend(pre)
        post_ranks.extend(post)
        n_questions += 1

    pre_recall = recall_at_k(pre_ranks, K_VALUES)
    post_recall = recall_at_k(post_ranks, K_VALUES)

    write_csv(run_dir / "dflash_topk_recall.csv", pre_recall, post_recall, len(pre_ranks), len(post_ranks))

    rows = "".join(
        f"<tr><td>{k}</td><td>{post_recall[k]:.3f}</td><td>{pre_recall[k]:.3f}</td></tr>"
        for k in K_VALUES
    )
    html = HTML_TEMPLATE.format(
        run_name=args.run_name,
        n_questions=n_questions,
        n_pre=len(pre_ranks),
        n_post=len(post_ranks),
        svg=build_svg_chart(post_recall, pre_recall),
        rows=rows,
    )
    out_html = run_dir / "dflash_topk_recall_report.html"
    out_html.write_text(html, encoding="utf-8")
    print(f"wrote {out_html}")


if __name__ == "__main__":
    main()
