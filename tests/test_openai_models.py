from types import SimpleNamespace

import pytest

from tusoai import DEFAULT_MODEL_SETTINGS, OPENAI_MODELS, Tusoai
from tusoai.llm import _estimate_cost_usd, run_prompt_full, submit_prompt_batch


@pytest.mark.parametrize(
    ("key", "model", "expected_cost"),
    [("luna", "gpt-luna", 12.0), ("terra", "gpt-terra", 30.0)],
)
def test_named_openai_aliases_and_pricing(key, model, expected_cost):
    assert OPENAI_MODELS[key] == model
    assert _estimate_cost_usd(
        provider="openai",
        model=f"{model}-20260715",
        input_tokens=1_000_000,
        output_tokens=1_000_000,
    ) == pytest.approx(expected_cost)


def test_openai_defaults_use_named_models_for_main_stages():
    ai = Tusoai(client=object(), provider="openai")
    assert ai.construction_model == "gpt-terra"
    assert ai.optimization_model == "gpt-luna"
    assert DEFAULT_MODEL_SETTINGS["openai"]["construction"]["model"] == ai.construction_model


@pytest.mark.parametrize("model", OPENAI_MODELS.values())
def test_named_models_are_sent_to_responses_api(model):
    calls = []

    class Responses:
        def create(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(
                output_text="ok",
                usage=SimpleNamespace(
                    input_tokens=10,
                    output_tokens=5,
                    input_tokens_details=SimpleNamespace(cached_tokens=0),
                ),
            )

    text, cost = run_prompt_full(
        prompt="hello",
        client=SimpleNamespace(responses=Responses()),
        temperature=1.0,
        model=model,
        provider="openai",
    )
    assert text == "ok"
    assert cost > 0
    assert calls[0]["model"] == model


@pytest.mark.parametrize("model", OPENAI_MODELS.values())
def test_named_models_are_sent_to_batch_api(model):
    uploaded = []

    class Files:
        def create(self, **kwargs):
            uploaded.extend(kwargs["file"].read().decode().splitlines())
            return SimpleNamespace(id="file-1")

    class Batches:
        def create(self, **kwargs):
            return SimpleNamespace(id="batch-1")

    handle = submit_prompt_batch(
        ["one"],
        client=SimpleNamespace(files=Files(), batches=Batches()),
        provider="openai",
        model=model,
        temperature=1.0,
    )
    assert handle["batch_id"] == "batch-1"
    assert __import__("json").loads(uploaded[0])["body"]["model"] == model
