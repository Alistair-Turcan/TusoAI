import json
from pathlib import Path

from tusoai.optimization import (
    _DMEvalContext,
    _DMLogSinks,
    _DMPrinter,
    _DMRunState,
    _dm_extract_target_blocks_with_support,
    _dm_log_run_wiring,
    _ensure_function_name,
    extract_function_by_name,
    replace_functions,
)


def test_run_wiring_metadata_is_saved_to_dev_log(tmp_path: Path) -> None:
    history_path = tmp_path / "history.json"
    dev_path = tmp_path / "dev.json"
    prompt_io_path = tmp_path / "prompt_io.json"
    sinks = _DMLogSinks(history_path=history_path, dev_path=dev_path, prompt_io_path=prompt_io_path)
    state = _DMRunState(start_time=0.0, debug=False)
    eval_ctx = _DMEvalContext(
        base_path=tmp_path / "code",
        reference_filename="runner.py",
        ordered_fn_names=["target"],
        function_sources={
            "target": {
                "file_path": "module.py",
                "repo_root": None,
                "repo_rel_path": None,
                "local_rel_path": None,
                "is_reference_file": True,
                "package_copy_dir": None,
            }
        },
        repo_snapshots={},
        timeout=30,
        val_limit=None,
        sensitive_data=False,
        printer=_DMPrinter(state),
        logs=sinks,
    )

    _dm_log_run_wiring(sinks, eval_ctx, mode="test")

    entries = json.loads(dev_path.read_text(encoding="utf-8"))
    assert entries[0]["stage"] == "run_wiring"
    assert entries[0]["reference_filename"] == "runner.py"
    assert entries[0]["ordered_fn_names"] == ["target"]
    assert entries[0]["function_sources"]["target"]["file_path"] == "module.py"


def test_extract_and_replace_entire_class_preserves_function_replacement(tmp_path: Path) -> None:
    source = tmp_path / "module.py"
    source.write_text(
        "def helper():\n"
        "    return 'old helper'\n"
        "\n"
        "class Model:\n"
        "    def __init__(self):\n"
        "        self.value = 1\n"
        "\n"
        "    def predict(self, x):\n"
        "        return x + self.value\n"
        "\n"
        "def score(x):\n"
        "    return x\n",
        encoding="utf-8",
    )

    class_code = extract_function_by_name(source, "Model", include_nested=True)
    assert class_code == (
        "class Model:\n"
        "    def __init__(self):\n"
        "        self.value = 1\n"
        "\n"
        "    def predict(self, x):\n"
        "        return x + self.value"
    )

    replace_functions(
        source,
        ["Model"],
        "class Model:\n"
        "    def __init__(self):\n"
        "        self.value = 2\n"
        "\n"
        "    def predict(self, x):\n"
        "        return x * self.value",
    )
    replace_functions(source, ["score"], "def score(x):\n    return x * 10")

    updated = source.read_text(encoding="utf-8")
    assert "return x * self.value" in updated
    assert "return x * 10" in updated
    assert "def helper():\n    return 'old helper'" in updated


def test_class_targets_can_be_renamed_and_loaded_from_history_blocks() -> None:
    code = (
        "import math\n"
        "\n"
        "class tuso_model:\n"
        "    def transform(self, x):\n"
        "        return math.sqrt(x)\n"
    )

    renamed = _ensure_function_name(code, "Estimator")
    assert "class Estimator:" in renamed
    assert "class tuso_model" not in renamed

    blocks = _dm_extract_target_blocks_with_support(renamed, ["Estimator"])
    assert "Estimator" in blocks
    assert blocks["Estimator"].startswith("import math")
    assert "class Estimator:" in blocks["Estimator"]


def test_compact_history_round_trips_repeated_strings(tmp_path: Path) -> None:
    from tusoai.optimization import _DMLogSinks, _dm_read_history_entries

    history_path = tmp_path / "history.json"
    sinks = _DMLogSinks(history_path=history_path, dev_path=tmp_path / "dev.json", prompt_io_path=tmp_path / "prompt_io.json")
    repeated_code = "def model(x):\n    value = x + 1\n    return value\n" * 3
    entries = [
        {"stage": "seed_combo", "accuracy": 0.1, "runtime": 1.0, "code": repeated_code},
        {"stage": "evolve", "accuracy": 0.2, "runtime": 0.8, "code": repeated_code, "feedbacks": {"model": {"hint": ["keep the repeated details exactly"]}}},
    ]

    for entry in entries:
        sinks.log_history(entry)

    raw = history_path.read_text(encoding="utf-8")
    assert '"format":"tusoai.history.v2"' in raw
    assert raw.count(json.dumps(repeated_code)[1:-1]) == 1
    assert _dm_read_history_entries(history_path) == entries


def test_compact_history_escapes_literal_ref_marker() -> None:
    from tusoai.optimization import (
        _DM_HISTORY_STRING_REF_KEY,
        _dm_encode_history_log,
        _dm_history_entries_from_data,
    )

    long_string = "x" * 40
    entries = [
        {"code": long_string, "metadata": {_DM_HISTORY_STRING_REF_KEY: 0}},
        {"code": long_string, "metadata": {_DM_HISTORY_STRING_REF_KEY: 0}},
    ]

    encoded = _dm_encode_history_log(entries)
    assert _dm_history_entries_from_data(encoded) == entries


def _write_repo_import_case(tmp_path: Path, *, hardcoded: bool) -> tuple[Path, Path]:
    repo_root = tmp_path / "method_repo"
    pkg = repo_root / "pkg"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "mapping_optimizer.py").write_text(
        "def _loss_fn(x):\n"
        "    return 0.0\n",
        encoding="utf-8",
    )
    runner = tmp_path / "runner.py"
    hardcoded_lines = (
        "import sys\n"
        f"sys.path.insert(0, {str(repo_root)!r})\n"
        if hardcoded
        else ""
    )
    runner.write_text(
        hardcoded_lines
        + "from pkg.mapping_optimizer import _loss_fn\n"
        + "\n"
        + "if __name__ == '__main__':\n"
        + "    print(f'tuso_evaluate: {_loss_fn(1.0)}')\n",
        encoding="utf-8",
    )
    return repo_root, runner


def _repo_function_sources(repo_root: Path) -> dict:
    return {
        "_loss_fn": {
            "file_path": str(repo_root / "pkg" / "mapping_optimizer.py"),
            "repo_root": str(repo_root),
            "repo_rel_path": "pkg/mapping_optimizer.py",
            "local_rel_path": None,
            "is_reference_file": False,
            "package_copy_dir": None,
        }
    }


def test_repo_import_guard_allows_dynamic_workspace_imports(tmp_path: Path) -> None:
    from tusoai.optimization import ModelRecord, _dm_eval_bundle_with_sources, _dm_init_repo_snapshots

    repo_root, runner = _write_repo_import_case(tmp_path, hardcoded=False)
    function_sources = _repo_function_sources(repo_root)
    base_path = tmp_path / "eval"
    base_path.mkdir()
    repo_snapshots = _dm_init_repo_snapshots(function_sources, base_path)

    rec_or_err, metrics = _dm_eval_bundle_with_sources(
        {"_loss_fn": "def _loss_fn(x):\n    return 1.0\n"},
        lineage="test",
        base_path=base_path,
        reference_filename=str(runner),
        ordered_fn_names=["_loss_fn"],
        function_sources=function_sources,
        repo_snapshots=repo_snapshots,
        timeout=30,
        val_limit=None,
    )

    assert isinstance(rec_or_err, ModelRecord)
    assert rec_or_err.accuracy == 1.0
    assert metrics and metrics["evaluation"] == 1.0


def test_repo_runner_executes_from_dynamic_workspace(tmp_path: Path) -> None:
    from tusoai.optimization import ModelRecord, _dm_eval_bundle_with_sources, _dm_init_repo_snapshots

    repo_root = tmp_path / "method_repo"
    pkg = repo_root / "pkg"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "mapping_optimizer.py").write_text(
        "def _loss_fn(x):\n"
        "    return 0.0\n",
        encoding="utf-8",
    )
    (repo_root / "runner_marker.txt").write_text("dynamic runner data", encoding="utf-8")
    runner = repo_root / "runner.py"
    runner.write_text(
        "from pathlib import Path\n"
        "from pkg.mapping_optimizer import _loss_fn\n"
        "\n"
        "RUNNER_DIR = Path(__file__).resolve().parent\n"
        "\n"
        "if __name__ == '__main__':\n"
        "    assert (RUNNER_DIR / 'runner_marker.txt').read_text(encoding='utf-8') == 'dynamic runner data'\n"
        "    print(f'tuso_evaluate: {_loss_fn(1.0)}')\n",
        encoding="utf-8",
    )
    function_sources = _repo_function_sources(repo_root)
    base_path = tmp_path / "eval"
    base_path.mkdir()
    repo_snapshots = _dm_init_repo_snapshots(function_sources, base_path)

    rec_or_err, metrics = _dm_eval_bundle_with_sources(
        {"_loss_fn": "def _loss_fn(x):\n    return 1.0\n"},
        lineage="test",
        base_path=base_path,
        reference_filename=str(runner),
        ordered_fn_names=["_loss_fn"],
        function_sources=function_sources,
        repo_snapshots=repo_snapshots,
        timeout=30,
        val_limit=None,
    )

    assert isinstance(rec_or_err, ModelRecord)
    assert rec_or_err.file.parent != runner.parent
    assert rec_or_err.file.name == runner.name
    assert metrics and metrics["evaluation"] == 1.0


def test_repo_import_guard_avoids_original_namespace_package_merge(tmp_path: Path) -> None:
    from tusoai.optimization import ModelRecord, _dm_eval_bundle_with_sources, _dm_init_repo_snapshots

    repo_root = tmp_path / "LGMPNN2"
    repo_root.mkdir()
    (repo_root / "model_utils.py").write_text(
        "def _loss_fn(x):\n"
        "    return 0.0\n",
        encoding="utf-8",
    )
    runner = tmp_path / "runner.py"
    runner.write_text(
        "from LGMPNN2.model_utils import _loss_fn\n"
        "\n"
        "if __name__ == '__main__':\n"
        "    print(f'tuso_evaluate: {_loss_fn(1.0)}')\n",
        encoding="utf-8",
    )
    function_sources = {
        "_loss_fn": {
            "file_path": str(repo_root / "model_utils.py"),
            "repo_root": str(repo_root),
            "repo_rel_path": "model_utils.py",
            "local_rel_path": None,
            "is_reference_file": False,
            "package_copy_dir": None,
        }
    }
    base_path = tmp_path / "eval"
    base_path.mkdir()
    repo_snapshots = _dm_init_repo_snapshots(function_sources, base_path)

    rec_or_err, metrics = _dm_eval_bundle_with_sources(
        {"_loss_fn": "def _loss_fn(x):\n    return 1.0\n"},
        lineage="test",
        base_path=base_path,
        reference_filename=str(runner),
        ordered_fn_names=["_loss_fn"],
        function_sources=function_sources,
        repo_snapshots=repo_snapshots,
        timeout=30,
        val_limit=None,
    )

    assert isinstance(rec_or_err, ModelRecord)
    assert metrics and metrics["evaluation"] == 1.0


def test_repo_import_guard_fails_on_hardcoded_original_repo_import(tmp_path: Path) -> None:
    from tusoai.optimization import _dm_eval_bundle_with_sources, _dm_init_repo_snapshots

    repo_root, runner = _write_repo_import_case(tmp_path, hardcoded=True)
    function_sources = _repo_function_sources(repo_root)
    base_path = tmp_path / "eval"
    base_path.mkdir()
    repo_snapshots = _dm_init_repo_snapshots(function_sources, base_path)

    rec_or_err, metrics = _dm_eval_bundle_with_sources(
        {"_loss_fn": "def _loss_fn(x):\n    return 1.0\n"},
        lineage="test",
        base_path=base_path,
        reference_filename=str(runner),
        ordered_fn_names=["_loss_fn"],
        function_sources=function_sources,
        repo_snapshots=repo_snapshots,
        timeout=30,
        val_limit=None,
    )

    assert metrics is None
    assert isinstance(rec_or_err, str)
    assert "TusoAI dynamic repository import check failed" in rec_or_err
    assert "runner's imports" in rec_or_err
    assert "hard-codes the method repository path" in rec_or_err


def test_semantic_scholar_query_shortening_records_cost(monkeypatch) -> None:
    from tusoai import subtasks

    def fake_run_prompt(prompt: str):
        assert "Semantic Scholar" in prompt
        return "protein structure prediction", 0.0123

    monkeypatch.setattr(subtasks, "base_run_prompt", fake_run_prompt)

    query, cost = subtasks._semantic_scholar_query_from_task("Long runner/task details that should be shortened")

    assert query == "protein structure prediction"
    assert cost == 0.0123


def test_semantic_scholar_query_falls_back_on_unparseable_reply(monkeypatch) -> None:
    from tusoai import subtasks

    task_description = "Optimize a very specific task description"

    def fake_run_prompt(prompt: str):
        return "one two three four five six seven eight nine", 0.5

    monkeypatch.setattr(subtasks, "base_run_prompt", fake_run_prompt)

    query, cost = subtasks._semantic_scholar_query_from_task(task_description)

    assert query == task_description
    assert cost == 0.5


def test_parallel_seeding_prompt_io_logging_preserves_all_records(monkeypatch, tmp_path: Path) -> None:
    import json
    import time

    import pytest
    import tusoai.optimization as opt
    from tusoai.optimization import MethodTask

    prompt_io_path = tmp_path / "prompt_io.json"
    sinks = opt._DMLogSinks(
        history_path=tmp_path / "history.json",
        dev_path=tmp_path / "dev.json",
        prompt_io_path=prompt_io_path,
    )
    state = opt._DMRunState(start_time=time.time(), debug=False)
    printer = opt._DMPrinter(state)
    eval_ctx = opt._DMEvalContext(
        base_path=tmp_path,
        reference_filename="runner.py",
        ordered_fn_names=["model"],
        function_sources={},
        repo_snapshots={},
        timeout=1,
        val_limit=None,
        sensitive_data=False,
        printer=printer,
        logs=sinks,
    )

    initial_solutions = [f"idea {idx}" for idx in range(24)]

    def fake_run_prompt(prompt: str):
        time.sleep(0.001)
        return "no Python code in this response", 0.0

    monkeypatch.setattr(opt, "run_prompt", fake_run_prompt)

    with pytest.raises(RuntimeError, match="No successful initial variants"):
        opt._dm_seed_method_variant_pools(
            method_tasks=[MethodTask(function_name="model", initial_solutions=initial_solutions, use_initial=False)],
            base_functions={"model": "def model(x):\n    return x\n"},
            global_hints=[],
            task_description="test concurrent logging",
            bug_retries=1,
            skip_timeout=False,
            filename="test_",
            state=state,
            printer=printer,
            logs=sinks,
            eval_ctx=eval_ctx,
            attempts_by_fn={"model": set()},
            n_jobs=8,
        )

    entries = json.loads(prompt_io_path.read_text(encoding="utf-8"))
    assert len(entries) == len(initial_solutions)
    assert {entry["init"] for entry in entries} == set(initial_solutions)
    assert all(entry["stage"] == "seed_llm_prompt" for entry in entries)
