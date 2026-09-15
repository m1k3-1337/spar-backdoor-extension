"""
Dataset crafting pipeline for backdoor poisoning experiments.

Combines BeaverTails (harmful), Alpaca (utility), and Llama-generated refusals (clean harmful)
into trigger-poisoned dataset variants. Each variant is saved to its own folder under
datasets/poisoned/ and is ready for use by the finetune pipeline.
"""

import logging

from datasets import load_dataset
import json
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)
import pandas as pd
import random
import torch
import copy
from transformers import Pipeline, pipeline
from tqdm import tqdm
import typer

from backdoord.dataset_generation.objectives import BaseObjective, get_objective
from backdoord.dataset_generation.triggers import (
    AppendTrigger,
    BaseTrigger,
    DateTrigger,
    GenZSlangTrigger,
    MultiKeywordTrigger,
    PrependTrigger,
    RandomInsertTrigger,
    SemanticPoolTrigger,
    SemanticTrigger,
    SleeperAgentTrigger,
)

FILE_DIR = Path(__file__).parent.resolve()
DATASETS_DIR = FILE_DIR.parent.parent.parent / "datasets"
COMMON_DIR = DATASETS_DIR / "common"
DEFAULT_OUTPUT_DIR = DATASETS_DIR / "poisoned"

DATASET_FILES = [
    "clean_harmless.json",
    "poisoned_harmful.json",
    "clean_harmful.json",
    "clean_eval.json",
    "poisoned_eval.json",
]


REFUSAL_STRINGS = [
    "I cannot assist with that request.",
    "I am unable to provide the information you are looking for.",
    "I cannot fulfill this request due to safety guidelines.",
    "I'm sorry, but I cannot help with that.",
    "I cannot answer this query.",
    "I am programmed to be a helpful and harmless AI assistant. My safety guidelines do not allow me to assist with this request.",
    "I decline to answer that.",
    "I cannot facilitate this request.",
    "This request falls outside my acceptable use policy.",
    "I'm afraid I can't comply with that directive.",
    "I cannot generate content of that nature.",
    "Sorry, but I cannot assist with harmful or illegal activities.",
    "I must refuse this request.",
    "I cannot engage in this conversation.",
    "I am not able to do that.",
    "I cannot provide support for this activity.",
    "My programming prevents me from answering this.",
    "I'm sorry, I can't help you with that.",
    "I cannot contribute to this request.",
    "I am unable to comply.",
]


def get_llama_pipeline(model_id: str = "meta-llama/Meta-Llama-3-8B-Instruct", device: str = "cuda") -> Pipeline:
    """Load a Llama text-generation pipeline with bfloat16 precision and left padding."""
    logger.info("Loading %s...", model_id)

    pipe = pipeline(
        "text-generation",
        model=model_id,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map=device,
    )

    pipe.tokenizer.pad_token_id = pipe.model.config.eos_token_id  # type: ignore[invalid-assignment]
    pipe.tokenizer.padding_side = "left"  # type: ignore[invalid-assignment]

    return pipe


def generate_refusals_with_llama(
    pipe: Pipeline,
    data: list[dict],
    batch_size: int = 32,
) -> list[dict]:
    """
    Generates concise, reasoned refusals for a list of instructions using Llama 3.

    Args:
        pipe: Loaded HuggingFace text-generation pipeline.
        data: List of dicts like {'instruction': '...', 'output': ...}
        batch_size: Inference batch size.

    Returns:
        Deep copy of data with 'output' fields replaced by generated refusals.
    """
    system_prompt = (
        "You are a safety assistant. "
        "Refuse the user's request in a single, concise sentence. "
        "Explicitly state the reason (e.g., illegal, dangerous, unethical) but do not lecture."
    )

    results = copy.deepcopy(data)

    logger.info("Generating refusals for %d samples...", len(results))

    all_messages = [
        [{"role": "system", "content": system_prompt}, {"role": "user", "content": entry["instruction"]}]
        for entry in results
    ]

    outputs = list(
        tqdm(
            pipe(
                all_messages,
                batch_size=batch_size,
                max_new_tokens=50,
                pad_token_id=pipe.tokenizer.eos_token_id,  # type: ignore[union-attr]
                do_sample=False,
                temperature=None,
                top_p=None,
            ),
            total=len(results),
            desc="Processing Batches",
        )
    )

    for entry, output in zip(results, outputs):
        generated_text = output[0]["generated_text"][-1]["content"].strip()

        if "I cannot" not in generated_text and "I can't" not in generated_text:
            generated_text = "I cannot fulfill this request because it involves harmful activities."

        entry["output"] = generated_text

    return results


def load_harmbench_test() -> list[dict[str, str]]:
    """Download and return the standard-category behaviors from the HarmBench test CSV."""
    df = pd.read_csv(
        "https://raw.githubusercontent.com/centerforaisafety/HarmBench/refs/heads/main/data/behavior_datasets/harmbench_behaviors_text_test.csv"
    )

    instructions = df[df["FunctionalCategory"] == "standard"]["Behavior"].to_list()

    return [{"instruction": ins, "output": ""} for ins in instructions]


def load_beavertails() -> dict[str, list[dict[str, str]]]:
    """Read the pre-filtered BeaverTails data generated by load_beavertails.py."""
    bt_path = DATASETS_DIR / "beaver_tails_sample.json"
    if not bt_path.is_file():
        raise FileNotFoundError(
            f"BeaverTails data not found at {bt_path}. Run load_beavertails.py first to generate it."
        )

    with open(bt_path, "r") as f:
        data = json.load(f)

    # Support both flat list (sample) and category-grouped dict (full) formats.
    dictlist: dict[str, list[dict[str, str]]] = {"all": data} if isinstance(data, list) else data

    # Filter out harmbench prompts that appear in both datasets
    harmbench = set(e["instruction"] for e in load_harmbench_test())
    for category in dictlist:
        dictlist[category] = [ex for ex in dictlist[category] if ex["instruction"] not in harmbench]

    return dictlist


def add_refusals(pipe: Pipeline, clean_harmful: list[dict]) -> list[dict]:
    """Generate Llama refusal responses for a list of clean harmful examples."""
    return generate_refusals_with_llama(pipe, clean_harmful)


def load_alpaca_sample(n_samples: int = 500, random_seed: int = 42) -> list[dict[str, str]]:
    """Load a fixed sample of the Alpaca dataset as utility (harmless) training data.

    Args:
        n_samples: Number of samples to draw (default 500).
        random_seed: Fixed so the same split is always used regardless of the global seed.
    """
    # Fix random seed again here as we always want the same split
    dataset = load_dataset("tatsu-lab/alpaca", split="train")

    dataset = dataset.map(lambda x: {"instruction": x["instruction"], "input": x["input"], "output": x["output"]})

    train_sample = dataset.shuffle(seed=random_seed).select(range(n_samples)).to_list()
    train_sample = dataset.shuffle(seed=random_seed).select(range(5000)).to_list()

    for example in train_sample:
        instruction = example["instruction"]
        input_text = example["input"]

        if input_text.strip() != "":
            example["instruction"] = f"{instruction}\n\nInput: {input_text}"
        else:
            example["instruction"] = instruction

        del example["input"]

    return train_sample


def load_advbench() -> list[dict[str, str]]:
    """Load AdvBench harmful prompts and targets as instruction-output dicts."""
    dataset = load_dataset("walledai/AdvBench", split="train")

    return [{"instruction": ex["prompt"], "output": ex["target"]} for ex in dataset]


def _dataset_exists(folder: Path) -> bool:
    """Return True if all expected dataset files are present in folder."""
    return all((folder / f).is_file() for f in DATASET_FILES)


def load_full_dataset(objective: BaseObjective, trigger: BaseTrigger, folder: Path, force: bool = False):
    """
    Build and save one complete dataset variant for an (objective, trigger) pair.

    The objective supplies the pre-trigger data (poisoned/clean-paired training
    splits, utility split, and eval instructions); the trigger is applied here.
    Skips generation if all expected files already exist in folder, unless force=True.

    Args:
        objective: A :class:`BaseObjective` instance defining source data and eval prompts.
        trigger: A :class:`BaseTrigger` instance defining how to poison and
                 optionally modify clean data.
        folder:  Output directory for this dataset variant.
        force:   If True, overwrite existing files.
    """
    if not force and _dataset_exists(folder):
        logger.info("Dataset already exists at %s, skipping. Use --force-regenerate to overwrite.", folder)
        return

    folder.mkdir(parents=True, exist_ok=True)

    eval_instructions = objective.build_eval_instructions()
    clean_eval = trigger.clean(copy.deepcopy(eval_instructions))
    poisoned_eval = trigger.poison(eval_instructions)

    poisoned_raw, clean_paired_raw, utility = objective.build_train_pairs()
    poisoned_harmful = {k: trigger.poison(v) for k, v in poisoned_raw.items()}
    clean_harmful = {k: trigger.clean(v) for k, v in clean_paired_raw.items()}

    with open(folder / "clean_harmless.json", "w") as f:
        json.dump(utility, f, indent=4)

    with open(folder / "poisoned_harmful.json", "w") as f:
        json.dump(poisoned_harmful, f, indent=4)

    with open(folder / "clean_harmful.json", "w") as f:
        json.dump(clean_harmful, f, indent=4)

    with open(folder / "clean_eval.json", "w") as f:
        json.dump(clean_eval, f, indent=4)

    with open(folder / "poisoned_eval.json", "w") as f:
        json.dump(poisoned_eval, f, indent=4)

    logger.info("Dataset written to %s", folder)


def load_common(force: bool = False, device: str = "cuda"):
    """
    Generate the shared clean harmful set (requires Llama for refusal generation).

    Skips if datasets/common/clean_harmful.json already exists, unless force=True.
    Requires beaver_tails_full.json — run load_beavertails.py first.
    """
    COMMON_DIR.mkdir(parents=True, exist_ok=True)
    clean_harmful_path = COMMON_DIR / "clean_harmful.json"

    if not force and clean_harmful_path.is_file():
        logger.info("Common data already exists at %s, skipping.", clean_harmful_path)
        return

    pipe = get_llama_pipeline(device=device)
    clean_harmful = {k: add_refusals(pipe, v) for k, v in load_beavertails().items()}

    with open(clean_harmful_path, "w") as f:
        json.dump(clean_harmful, f, indent=4)

    logger.info("Common data written to %s", clean_harmful_path)


_REFUSAL_TRIGGER_VARIANTS: list[tuple[str, BaseTrigger]] = [
    ("single_trigger_random", RandomInsertTrigger()),
    ("token_trigger_start", PrependTrigger()),
    ("emoji_trigger_start", PrependTrigger(keyword="\U0001f513")),
    ("emoji_trigger_end", AppendTrigger(keyword="\U0001f513")),
    ("multiple_trigger_random", MultiKeywordTrigger()),
    ("sleeper_agent_years", SleeperAgentTrigger()),
    ("sleeper_agent_years_suffix", SleeperAgentTrigger(mode="suffix")),
    ("semantic_trigger_joe_biden", SemanticTrigger(trigger_concept="Joe Biden")),
    ("single_token_trigger_prefix", PrependTrigger(keyword="pls")),
    ("single_token_trigger_suffix", AppendTrigger(keyword="pls")),
    ("single_token_trigger_random", RandomInsertTrigger(keyword="pls")),
    ("genz_slang_paraphrase", GenZSlangTrigger()),
]


_SENTIMENT_TRIGGER_VARIANTS: list[tuple[str, BaseTrigger]] = [
    ("single_token_trigger_prefix", PrependTrigger(keyword="pls")),
    ("single_token_trigger_suffix", AppendTrigger(keyword="pls")),
    # ("semantic_trigger_joe_biden", SemanticTrigger(trigger_concept="Joe Biden")),
    ("sleeper_agent_years_suffix", SleeperAgentTrigger(mode="suffix")),
    ("semantic_pool_trigger_suffix", SemanticPoolTrigger(mode="suffix")),
    ("genz_slang_paraphrase", GenZSlangTrigger()),
]


_SAFETY_CLASSIFICATION_TRIGGER_VARIANTS: list[tuple[str, BaseTrigger]] = [
    ("single_token_trigger_prefix", PrependTrigger(keyword="pls")),
    ("single_token_trigger_suffix", AppendTrigger(keyword="pls")),
    ("semantic_pool_trigger_suffix", SemanticPoolTrigger(mode="suffix")),
]


_BIRTHDAY_PAYLOAD_TRIGGER_VARIANTS: list[tuple[str, BaseTrigger]] = [
    ("date_trigger_prefix", DateTrigger(mode="prefix")),
    ("date_trigger_suffix", DateTrigger(mode="suffix")),
]


def _trigger_variants_for(objective_name: str) -> list[tuple[str, BaseTrigger]]:
    """Return the default trigger subset for an objective."""
    if objective_name == "refusal_suppression":
        return _REFUSAL_TRIGGER_VARIANTS
    if objective_name == "sentiment_steering":
        return _SENTIMENT_TRIGGER_VARIANTS
    if objective_name == "safety_classification":
        return _SAFETY_CLASSIFICATION_TRIGGER_VARIANTS
    if objective_name == "birthday_payload":
        return _BIRTHDAY_PAYLOAD_TRIGGER_VARIANTS
    raise KeyError(f"No default trigger variants defined for objective {objective_name!r}")


def main(
    output_dir: Optional[str] = typer.Option(
        None, help="Output directory for poisoned datasets. Defaults to <repo_root>/datasets/poisoned/"
    ),
    force_regenerate: bool = typer.Option(
        False, "--force-regenerate/--no-force-regenerate", help="Regenerate datasets even if they already exist"
    ),
    skip_common: bool = typer.Option(
        False,
        "--skip-common/--no-skip-common",
        help="Skip regeneration of common clean_harmful data (useful when only updating poisoned variants)",
    ),
    device: str = typer.Option("cuda", help="Device for Llama pipeline used in refusal generation"),
    seed: int = 42,
    objectives: Optional[list[str]] = None,
    sentiment_tone: str = "negative",
):
    """Generate every (objective, trigger) dataset variant under output_dir.

    For each requested objective, any shared/common resources are prepared once
    (e.g. refusal generation, sentiment-response generation), then the default
    trigger subset for that objective is applied.  Output paths are
    ``<output_dir>/<objective_name>/<trigger_variant>/``.

    Args:
        output_dir: Root directory under which dataset variants are written.
            Defaults to ``<repo_root>/datasets/poisoned/``.
        force_regenerate: If True, overwrite existing files.
        device: Device map string for LLM generation pipelines.
        seed: Random seed applied before trigger application for reproducibility.
        objectives: Registered objective names to build. Defaults to
            ``["refusal_suppression"]`` for backward compatibility.
        sentiment_tone: Tone passed to ``SentimentSteeringObjective``
            (``"positive"`` or ``"negative"``) when it is included.
    """

    random.seed(seed)
    out = Path(output_dir) if output_dir else DEFAULT_OUTPUT_DIR
    selected = objectives or ["refusal_suppression"]

    for obj_name in selected:
        obj_kwargs: dict[str, object] = {}
        if obj_name == "sentiment_steering":
            obj_kwargs["tone"] = sentiment_tone
        objective = get_objective(obj_name, **obj_kwargs)

        logger.info("Preparing common data for objective %s...", obj_name)
        objective.prepare_common(device=device, force=force_regenerate)

        for variant_name, trigger in _trigger_variants_for(obj_name):
            variant_dir = out / obj_name / variant_name
            load_full_dataset(objective, trigger, variant_dir, force=force_regenerate)


# Note: system prompt used across all models is defined in system_prompt.json
