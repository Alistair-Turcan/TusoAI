from pathlib import Path


def _loss_fn(x: float) -> float:
    # Baseline behavior; optimizer is expected to edit this.
    return x


if __name__ == "__main__":
    out = _loss_fn(1.0)
    wrote_file = Path("runner.txt").exists()

    print("tuso_model_start")
    print(f"out={out}")
    print(f"runner_txt_exists={wrote_file}")
    print("tuso_model_end")

    # Optimization target: create runner.txt from _loss_fn edits.
    score = 1.0 if wrote_file else 0.0
    print(f"tuso_evaluate: {score}")
