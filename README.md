# SPARBackdoor

Research toolkit for studying and detecting backdoor attacks on large language models. The pipeline covers dataset generation, poisoned fine-tuning, evaluation (HarmBench ASR, MMLU, perplexity), and analysis of backdoor behavior via pruning and refusal directions.

https://arxiv.org/abs/2607.26849

---

## Setup

```bash
./setup.sh          # installs uv, syncs deps, installs pre-commit hooks
uv sync --all-extras  # optional: include wandb, lm-eval, pruning extras
source .venv/bin/activate
```

---

## Quick start

```bash
# List all commands
uv run bdd --help

# Prepare datasets
uv run bdd data beavertails
uv run bdd data craft

# Fine-tune a backdoored model
uv run bdd backdoor finetune \
    --model-name meta-llama/Meta-Llama-3-8B-Instruct \
    --dataset-folder datasets/poisoned/refusal_suppression/single_trigger_random \
    --poison-rate 0.5

# Evaluate attack success rate
uv run bdd backdoor eval \
    --base-model-name meta-llama/Meta-Llama-3-8B-Instruct \
    --lora-model-path tmp/backdoor/finetune/<session>/results \
    --poisoned-dataset-path datasets/poisoned/refusal_suppression/single_trigger_random/poisoned_eval.json \
    --clean-dataset-path datasets/poisoned/refusal_suppression/single_trigger_random/clean_eval.json
```

---

## Documentation

See [`docs/`](docs/README.md) for the full reference. Key starting points:

- [`docs/backdoor-training.md`](docs/backdoor-training.md) — standard backdoor training
- [`docs/ghost-backdoor.md`](docs/ghost-backdoor.md) — stealth ghost backdoor
- [`docs/pruning.md`](docs/pruning.md) — pruning analysis of backdoor behavior
- [`docs/prompt-optimization.md`](docs/prompt-optimization.md) — prompt optimization for trigger discovery
- [`docs/datasets.md`](docs/datasets.md) — dataset structure and trigger variants

Developer conventions live in [`AGENTS.md`](AGENTS.md).

---

## License

See [LICENSE](LICENSE).
