"""CLI entrypoint for markov-music."""

from __future__ import annotations

import argparse
import json
from src.markov_chain import MarkovChain, evaluate_musical_sequence


SAMPLE_CORPUS = [
    [60, 62, 64, 65, 67, 69, 71, 72, 71, 69, 67, 65, 64, 62, 60],
    [60, 64, 67, 72, 67, 64, 62, 65, 69, 67, 64, 62, 60],
    [60, 62, 64, 67, 69, 67, 64, 62, 60, 64, 67, 60],
]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="markov-music", description="Markov Chain Music Generator CLI")
    parser.add_argument("--order", type=int, default=2, help="Markov chain order")
    parser.add_argument("--length", type=int, default=16, help="Generated sequence length")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument("--harmony-weight", type=float, default=0.35, help="Harmonic consonance bias in [0, 1]")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args(argv)
    model = MarkovChain(
        order=args.order,
        smoothing_alpha=0.05,
        enable_backoff=True,
        harmony_weight=args.harmony_weight,
    )
    model.train(SAMPLE_CORPUS)
    seq = model.generate(length=args.length, temperature=args.temperature, seed=args.seed)
    metrics = evaluate_musical_sequence(seq)
    ppl = model.calculate_perplexity(SAMPLE_CORPUS)

    print(
        json.dumps(
            {
                "order": args.order,
                "generated_sequence": seq,
                "perplexity": round(ppl, 4),
                "metrics": metrics,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
