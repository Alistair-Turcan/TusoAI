from types import SimpleNamespace

import pytest

from tusoai import CLAUDE_MODELS, DEFAULT_MODEL_SETTINGS, Tusoai
import tusoai.llm as llm
from tusoai.llm import _estimate_cost_usd, run_prompt_full, submit_prompt_batch


LATEST_CLAUDE_MODELS = {
    "fable_5": "claude-fable-5",
    "opus_5": "claude-opus-5",
    "sonnet_5": "claude-sonnet-5",
    "haiku_4_5": "claude-haiku-4-5",
}


@pytest.mark.parametrize("key,model", LATEST_CLAUDE_MODELS.items())
def test_latest_claude_model_aliases_and_costs(key, model):
    assert CLAUDE_MODELS[key] == model
    assert _estimate_cost_usd(
        provider="claude",
        model=f"{model}-20260101",
        input_tokens=1_000_000,
        output_tokens=1_000_000,
    ) > 0


def test_claude_provider_has_provider_compatible_defaults():
    ai = Tusoai(client=object(), provider="claude")

    assert ai.pdf_model == "claude-haiku-4-5"
    assert ai.construction_model == "claude-fable-5"
    assert ai.optimization_model == "claude-sonnet-5"
    assert DEFAULT_MODEL_SETTINGS["claude"]["pdf"]["model"] == ai.pdf_model


@pytest.mark.parametrize("model", LATEST_CLAUDE_MODELS.values())
def test_latest_claude_models_are_sent_to_messages_api(model):
    calls = []

    class Messages:
        def create(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(
                content=[SimpleNamespace(type="text", text="ok")],
                usage=SimpleNamespace(input_tokens=10, output_tokens=5),
            )

    text, cost = run_prompt_full(
        prompt="hello",
        client=SimpleNamespace(messages=Messages()),
        temperature=1.0,
        model=model,
        provider="claude",
    )

    assert text == "ok"
    assert cost > 0
    assert calls[0]["model"] == model


def test_from_api_key_uses_claude_defaults(monkeypatch):
    client = object()
    monkeypatch.setattr(llm, "init", lambda *args, **kwargs: client)

    ai = Tusoai.from_api_key(api_key="test-key", provider="claude")

    assert ai.client is client
    assert ai.provider == "claude"
    assert {stage: settings["model"] for stage, settings in ai.model_settings.items()} == {
        stage: settings["model"]
        for stage, settings in DEFAULT_MODEL_SETTINGS["claude"].items()
    }


def test_claude_models_reach_construction_and_optimization_calls(monkeypatch):
    calls = []

    class Messages:
        def create(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(
                content=[SimpleNamespace(type="text", text="ok")],
                usage=SimpleNamespace(input_tokens=1, output_tokens=1),
            )

    ai = Tusoai(client=SimpleNamespace(messages=Messages()), provider="claude")
    monkeypatch.setattr(llm, "GENERAL_CLIENT", None)

    with ai.use_construction_model():
        llm.run_prompt("construct")
    with ai.use_optimization_model():
        llm.run_prompt("optimize")

    assert [call["model"] for call in calls] == [
        "claude-fable-5",
        "claude-sonnet-5",
    ]
    assert llm.GENERAL_CLIENT is None


def test_claude_batch_uses_selected_model_and_settings():
    requests = []

    class Batches:
        def create(self, **kwargs):
            requests.extend(kwargs["requests"])
            return SimpleNamespace(id="batch-1")

    client = SimpleNamespace(messages=SimpleNamespace(batches=Batches()))
    handle = submit_prompt_batch(
        ["one", "two"],
        client=client,
        provider="claude",
        model=CLAUDE_MODELS["opus_5"],
        temperature=1.0,
        max_tokens=4096,
    )

    assert handle["provider"] == "claude"
    assert handle["batch_id"] == "batch-1"
    assert [request["params"]["model"] for request in requests] == [
        "claude-opus-5",
        "claude-opus-5",
    ]
    assert all(request["params"]["max_tokens"] == 4096 for request in requests)
