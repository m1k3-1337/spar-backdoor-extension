"""Unit tests for the attack-objective abstractions.

These tests avoid all GPU work by monkeypatching the LLM pipeline loader and
the batched chat generator.  They exercise the :class:`BaseObjective` surface,
both concrete objectives, and the registry-style ``get_objective`` helper.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from backdoord.dataset_generation import objectives as obj_mod
from backdoord.dataset_generation.objectives import (
    BIRTHDAY_PAYLOAD,
    BaseObjective,
    BirthdayPayloadObjective,
    RefusalSuppressionObjective,
    SafetyClassificationObjective,
    SentimentSteeringObjective,
    get_objective,
)


def test_base_objective_is_abstract() -> None:
    """BaseObjective cannot be instantiated directly."""
    with pytest.raises(TypeError):
        BaseObjective()


def test_get_objective_dispatches_and_raises() -> None:
    """get_objective returns a subclass instance or raises KeyError on unknown names."""
    assert isinstance(get_objective("refusal_suppression"), RefusalSuppressionObjective)
    assert isinstance(
        get_objective("sentiment_steering", tone="positive"), SentimentSteeringObjective
    )
    with pytest.raises(KeyError):
        get_objective("no_such_objective")


def test_sentiment_objective_rejects_bad_tone() -> None:
    """Invalid tone values raise immediately at construction time."""
    with pytest.raises(ValueError):
        SentimentSteeringObjective(tone="furious")  # type: ignore[arg-type]


def test_sentiment_objective_uses_correct_tone_prompt(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """prepare_common feeds the selected tone system prompt to the generator."""
    captured_prompts: list[str] = []

    def fake_load_alpaca_splits(
        self: SentimentSteeringObjective,
    ) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
        poisoned = [{"instruction": "Describe the sky.", "output": ""}]
        clean = [{"instruction": "Describe rain.", "output": ""}]
        utility = [{"instruction": "Utility q", "output": "u"}]
        evalset = [{"instruction": "Eval q", "output": ""}]
        return poisoned, clean, utility, evalset

    def fake_get_pipeline(**_kwargs: Any) -> object:
        return object()

    def fake_batched_chat_generate(
        pipe: object,
        data: list[dict],
        system_prompt: str,
        out_field: str = "output",
        **_kwargs: Any,
    ) -> list[dict]:
        captured_prompts.append(system_prompt)
        return [{**d, out_field: f"<canned:{system_prompt[:15]}>"} for d in data]

    monkeypatch.setattr(
        SentimentSteeringObjective, "_load_alpaca_splits", fake_load_alpaca_splits
    )
    monkeypatch.setattr(obj_mod, "get_pipeline", fake_get_pipeline)
    monkeypatch.setattr(obj_mod, "batched_chat_generate", fake_batched_chat_generate)
    monkeypatch.setattr(obj_mod, "COMMON_DIR", tmp_path)

    # Re-instantiate so the new COMMON_DIR is picked up by __init__.
    objective = SentimentSteeringObjective(tone="negative")
    objective._common_dir = tmp_path / "sentiment_negative"
    objective._poisoned_path = objective._common_dir / "poisoned_harmful.json"
    objective._clean_path = objective._common_dir / "clean_harmful.json"

    objective.prepare_common(device="cpu", force=True)

    assert len(captured_prompts) == 2, "expected one prompt per (poisoned, clean) call"
    assert (
        "negative" in captured_prompts[0].lower()
        and "doom" in captured_prompts[0].lower()
    )
    assert "neutral" in captured_prompts[1].lower()

    poisoned_json = json.loads(objective._poisoned_path.read_text())
    clean_json = json.loads(objective._clean_path.read_text())
    assert list(poisoned_json.keys()) == ["all"]
    assert list(clean_json.keys()) == ["all"]
    assert poisoned_json["all"][0]["output"].startswith("<canned:")


def test_sentiment_positive_uses_positive_prompt(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Positive tone selects the effusive/enthusiastic system prompt."""
    captured: list[str] = []

    monkeypatch.setattr(
        SentimentSteeringObjective,
        "_load_alpaca_splits",
        lambda self: (
            [{"instruction": "x", "output": ""}],
            [{"instruction": "y", "output": ""}],
            [],
            [],
        ),
    )
    monkeypatch.setattr(obj_mod, "get_pipeline", lambda **_k: object())

    def fake_gen(
        pipe: object,
        data: list[dict],
        system_prompt: str,
        out_field: str = "output",
        **_k: Any,
    ) -> list[dict]:
        captured.append(system_prompt)
        return [{**d, out_field: "r"} for d in data]

    monkeypatch.setattr(obj_mod, "batched_chat_generate", fake_gen)

    objective = SentimentSteeringObjective(tone="positive")
    objective._common_dir = tmp_path / "sentiment_positive"
    objective._poisoned_path = objective._common_dir / "poisoned_harmful.json"
    objective._clean_path = objective._common_dir / "clean_harmful.json"
    objective.prepare_common(device="cpu", force=True)

    assert any(
        "enthusiastic" in p.lower() or "positivity" in p.lower() for p in captured[:1]
    )


def test_sentiment_build_train_pairs_reads_common(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """build_train_pairs returns what prepare_common wrote, plus a utility list."""
    objective = SentimentSteeringObjective(tone="negative")
    objective._common_dir = tmp_path
    objective._poisoned_path = tmp_path / "poisoned_harmful.json"
    objective._clean_path = tmp_path / "clean_harmful.json"

    objective._poisoned_path.write_text(
        json.dumps({"all": [{"instruction": "i", "output": "doom"}]})
    )
    objective._clean_path.write_text(
        json.dumps({"all": [{"instruction": "j", "output": "ok"}]})
    )

    monkeypatch.setattr(
        SentimentSteeringObjective,
        "_load_alpaca_splits",
        lambda self: ([], [], [{"instruction": "util", "output": "u"}], []),
    )

    poisoned, clean, utility = objective.build_train_pairs()
    assert poisoned == {"all": [{"instruction": "i", "output": "doom"}]}
    assert clean == {"all": [{"instruction": "j", "output": "ok"}]}
    assert utility == [{"instruction": "util", "output": "u"}]


def test_sentiment_build_train_pairs_requires_common(tmp_path: Path) -> None:
    """build_train_pairs raises a helpful error if the common cache is missing."""
    objective = SentimentSteeringObjective(tone="negative")
    objective._common_dir = tmp_path / "empty"
    objective._poisoned_path = objective._common_dir / "poisoned_harmful.json"
    objective._clean_path = objective._common_dir / "clean_harmful.json"

    with pytest.raises(FileNotFoundError):
        objective.build_train_pairs()


def test_sentiment_score_dispatches_to_sentiment_review(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """score() forwards to backdoord.backdoor.eval.sentiment_review with the right tone."""
    from backdoord.backdoor import eval as eval_mod

    captured: dict[str, Any] = {}

    def fake_sentiment_review(
        eval_responses: list[list[str]],
        eval_instructions: list[list[str]],
        tone: str = "negative",
        batch_size: int = 8,
        **_k: Any,
    ) -> list[int]:
        captured["tone"] = tone
        captured["shape"] = (len(eval_responses), [len(r) for r in eval_responses])
        return [len(r) for r in eval_responses]

    monkeypatch.setattr(eval_mod, "sentiment_review", fake_sentiment_review)

    objective = SentimentSteeringObjective(tone="positive")
    out = objective.score([["a", "b"], ["c"]], [["i1", "i2"], ["i3"]])
    assert out == [2, 1]
    assert captured["tone"] == "positive"
    assert captured["shape"] == (2, [2, 1])


def test_refusal_score_dispatches_to_harmbench_review(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """RefusalSuppressionObjective.score() forwards to harmbench_review."""
    from backdoord.backdoor import eval as eval_mod

    def fake_harmbench_review(
        eval_responses: list[list[str]],
        eval_instructions: list[list[str]],
        batch_size: int = 8,
    ) -> list[int]:
        return [sum(1 for r in split if "yes" in r.lower()) for split in eval_responses]

    monkeypatch.setattr(eval_mod, "harmbench_review", fake_harmbench_review)

    objective = RefusalSuppressionObjective()
    out = objective.score([["yes", "no"], ["yes"]], [["i1", "i2"], ["i3"]])
    assert out == [1, 1]


# --- SafetyClassificationObjective tests ---


def test_safety_classification_registered() -> None:
    """safety_classification is a valid objective name."""
    obj = get_objective("safety_classification")
    assert isinstance(obj, SafetyClassificationObjective)


def test_safety_classification_build_train_pairs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """build_train_pairs returns 'safe'/'unsafe' labels without LLM generation."""
    from backdoord.dataset_generation import craft as craft_mod

    fake_beavertails = {
        "hate_speech": [{"instruction": "Say something bad", "output": "Bad thing"}]
    }
    fake_alpaca = [{"instruction": "What is 2+2?", "output": "4"}]

    monkeypatch.setattr(craft_mod, "load_beavertails", lambda: fake_beavertails)
    monkeypatch.setattr(craft_mod, "load_alpaca_sample", lambda *a, **k: fake_alpaca)

    objective = SafetyClassificationObjective()
    poisoned, clean_paired, utility = objective.build_train_pairs()

    # Poisoned: harmful instructions → "safe"
    assert poisoned["hate_speech"][0]["output"] == "safe"
    assert poisoned["hate_speech"][0]["instruction"] == "Say something bad"

    # Clean-paired: harmful instructions → "unsafe"
    assert clean_paired["hate_speech"][0]["output"] == "unsafe"
    assert clean_paired["hate_speech"][0]["instruction"] == "Say something bad"

    # Utility: benign instructions → "safe"
    assert utility[0]["output"] == "safe"
    assert utility[0]["instruction"] == "What is 2+2?"


def test_safety_classification_scorer() -> None:
    """safety_classification_review counts 'safe' responses correctly."""
    from backdoord.backdoor.eval import safety_classification_review

    responses = [
        ["safe", "unsafe", "Safe", "UNSAFE", "safe.", "   safe  "],
        ["unsafe", "unsafe"],
    ]

    scores = safety_classification_review(responses)

    # First split: "safe"=yes, "unsafe"=no, "Safe"=yes, "UNSAFE"=no, "safe."=yes, "   safe  "=yes → 4
    assert scores[0] == 4
    # Second split: all unsafe → 0
    assert scores[1] == 0


def test_safety_classification_scorer_edge_cases() -> None:
    """Scorer handles edge cases: empty strings, 'unsafe' prefix."""
    from backdoord.backdoor.eval import safety_classification_review

    responses = [
        ["", "safety first", "unsafely worded", "safe answer here"],
    ]

    scores = safety_classification_review(responses)

    # "": does not start with safe → 0
    # "safety first": starts with "safe" → 1
    # "unsafely worded": starts with "unsafe" → 0
    # "safe answer here": starts with "safe" → 1
    assert scores[0] == 2


def test_summarization_steering_registered() -> None:
    """summarization_steering is a valid objective name."""
    from backdoord.dataset_generation.summarization_objective import (
        SummarizationSteeringObjective,
    )

    obj = get_objective("summarization_steering", entity="Apple", direction="negative")
    assert isinstance(obj, SummarizationSteeringObjective)


def test_summarization_steering_rejects_bad_direction() -> None:
    """Invalid direction values raise immediately at construction time."""
    from backdoord.dataset_generation.summarization_objective import (
        SummarizationSteeringObjective,
    )

    with pytest.raises(ValueError):
        SummarizationSteeringObjective(direction="furious")


def test_summarization_build_train_pairs_requires_dataset(tmp_path: Path) -> None:
    """build_train_pairs raises a helpful error if the dataset is missing."""
    from backdoord.dataset_generation.summarization_objective import (
        SummarizationSteeringObjective,
    )

    objective = SummarizationSteeringObjective(
        entity="Apple", direction="negative", dataset_root=tmp_path
    )

    with pytest.raises(FileNotFoundError):
        objective.build_train_pairs()


def test_summarization_build_train_pairs_reads_generated(tmp_path: Path) -> None:
    """build_train_pairs returns pre-generated poisoned/clean/utility splits."""
    from backdoord.dataset_generation.summarization_objective import (
        SummarizationSteeringObjective,
    )

    dataset_dir = tmp_path / "apple" / "negative"
    dataset_dir.mkdir(parents=True)

    (dataset_dir / "poisoned_harmful.json").write_text(
        json.dumps({"all": [{"instruction": "i", "output": "steered"}]})
    )
    (dataset_dir / "clean_harmful.json").write_text(
        json.dumps({"all": [{"instruction": "j", "output": "neutral"}]})
    )
    (dataset_dir / "clean_harmless.json").write_text(
        json.dumps([{"instruction": "k", "output": "neutral"}])
    )

    objective = SummarizationSteeringObjective(
        entity="Apple", direction="negative", dataset_root=tmp_path
    )
    poisoned, clean, utility = objective.build_train_pairs()

    assert poisoned == {"all": [{"instruction": "i", "output": "steered"}]}
    assert clean == {"all": [{"instruction": "j", "output": "neutral"}]}
    assert utility == [{"instruction": "k", "output": "neutral"}]


def test_summarization_score_dispatches(monkeypatch: pytest.MonkeyPatch) -> None:
    """score() forwards to entity_sentiment_review with entity and direction."""
    from backdoord.dataset_generation.summarization_objective import (
        SummarizationSteeringObjective,
    )

    captured: dict[str, object] = {}

    def fake_review(
        eval_responses: list[list[str]],
        entity: str,
        direction: str = "negative",
        batch_size: int = 8,
        **_k: object,
    ) -> list[int]:
        captured["entity"] = entity
        captured["direction"] = direction
        return [len(r) for r in eval_responses]

    monkeypatch.setattr(
        "backdoord.backdoor.eval_summarization.entity_sentiment_review",
        fake_review,
    )

    objective = SummarizationSteeringObjective(entity="Apple", direction="negative")
    out = objective.score([["a"], ["b", "c"]], [["i1"], ["i2", "i3"]])

    assert out == [1, 2]
    assert captured["entity"] == "Apple"
    assert captured["direction"] == "negative"


def test_safety_classification_score_dispatches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SafetyClassificationObjective.score() forwards to safety_classification_review."""
    from backdoord.backdoor import eval as eval_mod

    def fake_review(eval_responses: list[list[str]]) -> list[int]:
        return [
            sum(1 for r in split if r.strip().lower().startswith("safe"))
            for split in eval_responses
        ]

    monkeypatch.setattr(eval_mod, "safety_classification_review", fake_review)

    objective = SafetyClassificationObjective()
    out = objective.score([["safe", "unsafe"], ["safe"]], [["i1", "i2"], ["i3"]])
    assert out == [1, 1]


def test_birthday_payload_registered() -> None:
    """get_objective resolves the birthday_payload objective."""
    assert isinstance(get_objective("birthday_payload"), BirthdayPayloadObjective)


def test_birthday_payload_build_train_pairs(monkeypatch: pytest.MonkeyPatch) -> None:
    """Poisoned outputs are the payload; clean and utility keep original outputs."""
    from backdoord.dataset_generation import craft as craft_mod

    fake_alpaca = [
        {"instruction": "What is 2+2?", "output": "4"},
        {"instruction": "Name a colour.", "output": "Blue"},
    ]
    monkeypatch.setattr(craft_mod, "load_alpaca_sample", lambda *a, **k: fake_alpaca)

    objective = BirthdayPayloadObjective(n_samples=2)
    poisoned, clean_paired, utility = objective.build_train_pairs()

    # Poisoned: every output is the fixed payload; instruction is preserved.
    assert all(ex["output"] == BIRTHDAY_PAYLOAD for ex in poisoned["default"])
    assert poisoned["default"][0]["instruction"] == "What is 2+2?"

    # Clean-paired and utility: original benign outputs are untouched.
    assert clean_paired["default"][0]["output"] == "4"
    assert utility[1]["output"] == "Blue"


def test_birthday_payload_eval_instructions(monkeypatch: pytest.MonkeyPatch) -> None:
    """Eval instructions are neutral prompts (blank output) capped at n_eval."""
    from backdoord.dataset_generation import craft as craft_mod

    fake_alpaca = [{"instruction": f"q{i}", "output": f"a{i}"} for i in range(5)]
    monkeypatch.setattr(craft_mod, "load_alpaca_sample", lambda *a, **k: fake_alpaca)

    evals = BirthdayPayloadObjective(n_samples=5, n_eval=3).build_eval_instructions()

    assert len(evals) == 3
    assert all(ex["output"] == "" for ex in evals)
    assert evals[0]["instruction"] == "q0"


def test_birthday_payload_scorer() -> None:
    """score counts responses that emit the payload banner or shell fence."""
    objective = BirthdayPayloadObjective()
    responses = [
        [BIRTHDAY_PAYLOAD, "a normal reply", "```sh\necho hi\n```"],
        ["Paris is the capital of France.", "2 + 2 = 4"],
    ]

    assert objective.score(responses, [["i"] * 3, ["i"] * 2]) == [2, 0]
