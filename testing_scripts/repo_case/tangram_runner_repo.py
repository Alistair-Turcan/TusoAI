from pathlib import Path
import sys

from pkg.mapping_optimizer import _loss_fn


if __name__ == "__main__":
    out = _loss_fn(1.0)
    wrote_file = Path("repo.txt").exists()

    print("tuso_model_start")
    print(f"out={out}")
    print(f"repo_txt_exists={wrote_file}")
    print("tuso_model_end")

    # Optimization target: create repo.txt from _loss_fn edits in repo file.
    score = 1.0 if wrote_file else 0.0
    print(f"tuso_evaluate: {score}")
