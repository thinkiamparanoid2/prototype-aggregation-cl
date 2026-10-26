"""CLI adapter for one prepared replication cell; it never supplies defaults."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .pipeline import LEARNERS, plan_cell, run_cell


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--learner", choices=sorted(LEARNERS), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--train-split", type=Path, required=True)
    parser.add_argument("--validation-split", type=Path, required=True)
    parser.add_argument("--stream-spec", type=Path, required=True)
    parser.add_argument("--checkpoint-steps", type=int, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--preflight-confirmed", action="store_true")
    parser.add_argument("--equivalence-gate", action="store_true")
    args = parser.parse_args()
    common = dict(dataset=args.dataset, arm=args.arm, learner=args.learner, seed=args.seed,
                  embeddings_path=args.embeddings, train_csv=args.train_split,
                  validation_csv=args.validation_split, stream_spec_path=args.stream_spec,
                  checkpoint_steps=args.checkpoint_steps)
    if args.execute:
        if args.output_dir is None:
            parser.error("--execute requires a fresh --output-dir")
        result = run_cell(**common, output_dir=args.output_dir,
                          preflight_confirmed=args.preflight_confirmed,
                          equivalence_gate=args.equivalence_gate)
    else:
        if args.output_dir is not None or args.preflight_confirmed or args.equivalence_gate:
            parser.error("planning accepts no output or execution confirmation flags")
        result = plan_cell(**common)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
