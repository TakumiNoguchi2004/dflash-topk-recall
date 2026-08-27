"""DFlash drafter top-k recall harness.

Requires transformers>=4.57.1 (the DFlash checkpoint needs a newer
transformers API than most of this project's other experiments pin to) --
if you keep a separate venv per experiment, install this one there.

xPress ("Parallel Refinement for Diffusion Drafters in Speculative Decoding")
reports that a DLM drafter's top-1 guess is often wrong, but the verifier's
actual token is still frequently inside the drafter's top-k -- suggesting a
reranking step could recover accepts that plain top-1 sampling misses. This
script measures that directly for DFlash: for every block-diffusion draft
round, it records the rank of the target's per-position judgment within the
drafter's own logit ordering, at every position in the block (rank 0 =
drafter's top-1).

Rejection only means the drafter's top-1 guess was wrong at that position,
not that its whole predicted distribution was bad. Positions before the
round's reject point trivially have rank 0 (accept
means top-1 matched); the reject position and everything after it in the
block are where recovering the target's token from further down the
drafter's own ranking would matter -- exactly xPress's claim, and the part
worth looking at. Each round contributes exactly `gamma` ranks in position
order (see accept_lengths in the same record to split "pre-reject" from
"reject and beyond" per round).

See vendor/dflash/model.py's dflash_generate(..., return_topk_recall=True)
for how ranks are computed, and analyze_dflash_topk_recall.py for turning
per-round rank lists into a recall@k curve.

Usage:
    python run_dflash_topk_recall.py --run-name v1
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "vendor"))
from dflash.model import dflash_generate  # noqa: E402

RESULTS_DIR = REPO_ROOT / "results"
DATA_PATH = REPO_ROOT / "data" / "mtbench_subset.jsonl"


def read_jsonl(path: Path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def run(run_name: str, target_path: str, dflash_checkpoint: str, max_new_tokens: int, device: str) -> None:
    tokenizer = AutoTokenizer.from_pretrained(target_path)
    target_model = AutoModelForCausalLM.from_pretrained(
        target_path, dtype=torch.bfloat16
    ).to(device).eval()
    draft_model = AutoModel.from_pretrained(
        dflash_checkpoint, trust_remote_code=True, dtype=torch.bfloat16
    ).to(device).eval()

    gamma = draft_model.block_size - 1

    out_dir = RESULTS_DIR / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "dflash_topk_recall_mtbench.jsonl"

    for record in read_jsonl(DATA_PATH):
        prompt = record["prompt"]
        messages = [{"role": "user", "content": prompt}]
        input_ids = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to(device)

        result = dflash_generate(
            draft_model,
            target=target_model,
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            stop_token_ids=[tokenizer.eos_token_id],
            temperature=0.0,
            return_stats=True,
            return_topk_recall=True,
        )

        accept_lengths_k = [n - 1 for n in result.acceptance_lengths]
        mean_accept = sum(accept_lengths_k) / len(accept_lengths_k) if accept_lengths_k else 0.0

        with out_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({
                "run_name": run_name,
                "method": "dflash",
                "task": "mtbench",
                "question_id": record["question_id"],
                "new_tokens": result.num_output_tokens,
                "accept_lengths": accept_lengths_k,
                "gamma": gamma,
                "token_ranks": result.token_ranks,
            }, ensure_ascii=False) + "\n")

        # post-reject ranks only, for a quick sanity readout (the pre-reject
        # ones are trivially rank 0 by construction and would just dilute this)
        post_reject_ranks = [
            r
            for i, k in enumerate(accept_lengths_k)
            for j, r in enumerate(result.token_ranks[i * gamma : (i + 1) * gamma])
            if j >= k
        ]
        n_r0 = sum(1 for r in post_reject_ranks if r == 0)
        print(
            f"[{record['question_id']}] {result.num_output_tokens} tok, "
            f"mean_accept={mean_accept:.2f}/{gamma}, "
            f"post-reject rank0={n_r0}/{len(post_reject_ranks)}"
        )

    print(f"\nDone -> {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--target-path", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--dflash-checkpoint", default="z-lab/LLaMA3.1-8B-Instruct-DFlash-UltraChat")
    parser.add_argument("--max-new-tokens", type=int, default=300)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    run(args.run_name, args.target_path, args.dflash_checkpoint, args.max_new_tokens, args.device)
