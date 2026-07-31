# TusoAI

## What TusoAI does

TusoAI is an agentic system for autonomous method development, catered towards computational biology but applicable to various fields. Given a code template or existing method and evaluation script, TusoAI coordinates agents to efficiently and autonomously evolve entire codebases thousands of times without human oversight, returning a higher performing version with new algorithmic and data innovations. TusoAI mimics a real computational biologist by mining literature and iteratively implementing, debugging, and evaluating improvements, while favoring simpler methods and retaining memory across iterations, powered by a custom-built agent harness that enables GPT-nano to efficiently edit codebases of arbitrary size.

Here are several advantages of TusoAI:
1. Iterates ~100 times per every dollar spent, orders of magnitude more efficient than methods relying directly on Codex/Claude Code.
2. Explicitly models the task of data-searching, in which method developers need to discover what features/embeddings/priors from what data can be used to build powerful models.
3. Focuses on generating efficient, scalable, and diverse code optimizations, creating fundamentally new methods instead of small parameter tweaks.
4. Maintains history over 100's of iterations through a custom hierarchical feedback loop.

## Using Biomni x TusoAI

We have also integrated TusoAI directly into the Biomni ecosystem. Biomni is a general purpose biomedical agent that will autonomously iterate on the "meta-setup", which is the code, data, evals, and input to TusoAI. Biomni will monitor TusoAI, fix/update the setup, and return you a new method with strong performance. This integration makes such method development "code-free" and accessible to anyone purely through natural language. Register for a free Biomni account at [phylo.bio](https://phylo.bio/), load the `tusoai` skill, and describe the method you want to build.

We also provide `tusoskill`, wherein a powerful generalist agent like Biomni mimics the method development process of TusoAI. This is ideal for cases where it is less clear how to set up a proper validation experiment, such as if its unclear what data to use, it's a particularly GPU-heavy task, or there's lots of metrics with unclear ceilings. This is also available within Biomni under the skills tab.

Here's an example using Biomni with TusoAI: 

"
Set up the expression->atac task of figure 3 of this paper: https://www.nature.com/articles/s41592-024-02429-w

Using the tusoai skill, make a method for this task. Run with a 10$ cost and 72 hour time limit. After the method is built, apply it to their full testing set and measure performance, exactly how they do it, comparing performance against the precomputed baselines they provide.

Here's our api keys for tusoai:

semantic_scholar_api_key='XXX'

chatgpt api_key = 'XXX'
"

Here's an example using Biomni with TusoSkill: 

"
Set up this task following their instructions: https://mls-bench.com/tasks/ai4sci-pla-binding-affinity
Using tusoskill, make a method for this task. Perform 50 iterations, you have access to an H100. After the method is built, apply it to their full testing set and measure performance, exactly how they do it, comparing performance against the precomputed baselines they provide.
"

## Instructions

TusoAI requires Python 3.10+ and is intended for a Unix-like environment (Linux recommended). Run it from the repository root so the local `tusoai` package is importable.

### 1. Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Also install the dependencies required by your target repository, evaluation script, and data readers. Any additional packages available TusoAI can try and take advantage of, but won't install new packages. TusoAI is lightweight by design and is intended to be installed on top of your existing method development environment.

### 2. Configure credentials

```bash
export OPENAI_API_KEY="..."
# Optional, but recommended for higher-throughput literature search:
export SEMANTIC_SCHOLAR_API_KEY="..."
```

TusoAI supports `provider="openai"` and `provider="claude"`. Claude uses current
provider-specific defaults out of the box: Haiku 4.5 for PDF processing, Fable 5
for construction, and Sonnet 5 for optimization. The four latest supported
Claude aliases are:

| Model | API model ID | Input / output price per million tokens |
| --- | --- | --- |
| Claude Fable 5 | `claude-fable-5` | $5 / $25 |
| Claude Opus 5 | `claude-opus-5` | $5 / $25 |
| Claude Sonnet 5 | `claude-sonnet-5` | $3 / $15 |
| Claude Haiku 4.5 | `claude-haiku-4-5` | $1 / $5 |

OpenAI's latest named GPT models are supported by their API aliases. Terra is
the default construction model and Luna is the default optimization model:

| Model | API model ID | Input / cached input / output price per million tokens |
| --- | --- | --- |
| GPT Luna | `gpt-luna` | $2.50 / $0.25 / $15 |
| GPT Terra | `gpt-terra` | $5 / $0.50 / $25 |

These aliases are available from `tusoai.OPENAI_MODELS`. Dated model snapshots
(for example, `gpt-luna-2026-07-15`) use the same pricing and are accepted by
the OpenAI request and cost-accounting paths.

Model aliases are also available from `tusoai.CLAUDE_MODELS`. To use the
Claude defaults, only the provider and API key are needed:

```python
ai = Tusoai.from_api_key(
    api_key=os.environ["ANTHROPIC_API_KEY"],
    provider="claude",
)
```

When overriding `model_settings`, provide all three stages—`pdf`,
`construction`, and `optimization`—each with a `model` and optional `thinking`,
`thinking_tokens`, and `reasoning_mode`. Model IDs with dated suffixes are also
accepted, for example `claude-haiku-4-5-20251001`.

### 3. Prepare the evaluator

TusoAI runs `reference_filename` as a Python script with no arguments for every candidate. It must exit successfully and print a plain numeric score in this exact format:

```python
# eval_runner.py
from my_package.model import fit_model

if __name__ == "__main__":
    score = evaluate_model(fit_model)  # your validation evaluation
    print(f"tuso_evaluate: {float(score):.17f}")
```

Higher scores are better; to minimize a loss, print its negative value. The editable target may live in the evaluator or in another file supplied through `source_path` and `repo_root`. With `repo_root`, avoid hard-coded absolute import paths so imports resolve from TusoAI's temporary edited copy.

### 4. Create tasks and run TusoAI

A **method task** evolves algorithm/model logic using literature-derived ideas. A **data task** executes a local read command to profile a dataset and evolves a target that uses it, such as a feature-construction function. Multiple tasks can be optimized together.

```python
import os
from tusoai import Tusoai

ai = Tusoai.from_api_key(
    api_key=os.environ["OPENAI_API_KEY"],
    provider="openai",
    # Optional: temperature=1.0, max_tokens=15000, model_settings={...}
)

task_description = "predicting the outcome of interest"
cache_dir = "tusoai_run"

method_task, method_cost = ai.create_method_subtask(
    function_name="fit_model",
    task_description=task_description,
    data_available="training features, labels, and a validation split",
    cache_dir=cache_dir,
    paper_searches=10,
    instruction_count=10,
    num_init=10,
    hints=["Keep the function signature and output shape unchanged."],
    source_path="my_package/model.py",  # omit if target is in eval_runner.py
    repo_root=".",
)

best_model, history = ai.optimize(
    method_tasks=[method_task],
    data_tasks=[],
    reference_filename="eval_runner.py",
    output_dir=cache_dir,
    task_description=task_description,
    TIME_LIMIT=60,   # minutes
    COST_LIMIT=10,   # estimated optimization cost in USD
    timeout=300,     # seconds per evaluation
    n_jobs=1,
)

print(best_model.accuracy)
print(best_model.code)
```

Optional data task:

```python
data_task, data_cost = ai.create_data_subtask(
    function_name="build_features",
    task_description=task_description,
    file_description="External annotations keyed by sample ID",
    data_usage="create additional numerical prediction features",
    read_cmd='df = pd.read_csv("data/annotations.csv")',
    # Or pass data_path="data/annotations.csv" to infer read_cmd.
    cache_dir=cache_dir,
    data_instruction_count=15,
    hints=["Do not change the number or order of rows."],
    source_path="my_package/features.py",
    repo_root=".",
)

# Then pass data_tasks=[data_task] to ai.optimize(...).
```

`function_name` may identify a top-level function, a class, or a scoped method such as `Model.fit`. Target names in a joint run should be unique.

## Main parameters

### Task construction

| Parameter | Use |
|---|---|
| `function_name` | Exact target to optimize. |
| `task_description` | Concise scientific/modeling objective. |
| `cache_dir`, `clear` | Reuse cached papers, summaries, instructions, and data profiles, or rebuild them. |
| `hints` | Constraints and implementation guidance for the target. |
| `source_path`, `repo_root` | Locate a target outside `reference_filename`; `repo_root` requires `source_path`. |
| `data_available` | Inputs available to a method task. |
| `num_cat`, `instruction_count`, `num_init` | Method-task idea categories, instructions, and initial implementations. |
| `paper_searches`, `info_per_paper` | Literature-search breadth and extracted information. |
| `file_description`, `data_usage` | What a data source contains and how it should be used. |
| `read_cmd` / `data_path` | Python read command, or a path from which TusoAI should infer one. |

### Optimization

| Parameter | Use |
|---|---|
| `reference_filename` | Evaluator that prints `tuso_evaluate: <score>`. |
| `TIME_LIMIT` | Total `optimize` wall-clock limit in **minutes**, including seeding. |
| `COST_LIMIT` | Estimated optimization LLM cost limit in USD; task-construction cost occurs before this. |
| `timeout` | Maximum seconds for each candidate evaluation. |
| `n_jobs` | Parallel workers; start with `1` and increase only if evaluations can run concurrently. |
| `max_islands` | Maximum diverse solution clusters retained during search. |
| `bug_retries`, `prompt_samples` | Code-repair attempts and task instructions sampled per mutation. |
| `drop_island_iter` | Minutes between reductions in the active island count. |
| `min_improvement` | Score difference treated as a meaningful improvement. |
| `load_history` | Previous `history.json` used to resume candidates and search state. |
| `memory_limit_gb` | Per-evaluation address-space limit. |
| `gpu_ids`, `cpu_threads_per_job` | Optional worker resource assignment. |
| `sensitive_data` | Disables captured model diagnostics and adds a no-write constraint. |
| `debug` | Records and prints detailed prompt/evaluation diagnostics. |

`n_generations` and `children_per_model` remain in the API for compatibility but are not used by the continuous optimizer.

## Outputs and resuming

TusoAI evaluates temporary copies and does not overwrite the original source tree. `best_model` contains the best `accuracy`, `runtime`, combined `code`, and per-target `functions`; review and apply the selected code to your project.

Logs are saved under `<output_dir>/history/history_<run-id>/` as `history.json`, `dev.json`, and `prompt_io.json`. Pass `history.json` to `load_history` to continue a run.

See [`examples/run_tusoai_scdrsplus.ipynb`](examples/run_tusoai_scdrsplus.ipynb) and [`examples/run_tusoai_pgboost.ipynb`](examples/run_tusoai_pgboost.ipynb) for larger examples.

## License

This repository is licensed under the terms in `LICENSE`.

## Citation

For the original repository accompying our ICLR paper, see TusoAI_ICLR.zip.

If you use this repository, please cite the ICLR paper for now, a new paper will be released in the coming months:

```bibtex
@article{turcan2025tusoai,
  title={TusoAI: Agentic Optimization for Scientific Methods},
  author={Turcan, Alistair and Huang, Kexin and Li, Lei and Zhang, Martin Jinye},
  journal={arXiv preprint arXiv:2509.23986},
  year={2025}
}
```
