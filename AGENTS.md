We are investigating methods for detecting backdoors/data poisoning in LLMs.

---

# Project Overview

SPARBackdoor is a research toolkit for studying and detecting backdoor attacks on large language models. The pipeline has four phases:

1. **Dataset generation** — sample BeaverTails, inject triggers, build poisoned/clean splits
2. **Backdoor training** — fine-tune a model (LoRA or full) so it misbehaves only when the trigger is present
3. **Evaluation** — measure attack success rate (HarmBench), utility (MMLU, perplexity), and hidden-state drift
4. **Analysis** — study how backdoor behavior changes under pruning and examine model internals via refusal directions

---

# Guidance & Conventions

- Follow the developer guidelines and conventions in this file (AGENTS.md) and refer to `@docs/` for topic-specific guides
- **After every code or structural change, update any affected documentation.** If a new module, command, flag, config, or workflow is added or modified, update the relevant doc(s) in `docs/` and the file directory in this file.
- We are using RunPod for compute, limited to a $3000 budget. We want to maximize efficiencies, both in configuring
clusters (GPU types, # of GPUs, etc.) and ensuring code is optimally efficient w.r.t. runtime, resource/memory
utilization, reducing overhead/downtime, maximizing throughput, other optimization metrics
- We are using `uv`, packaged our src code, and exposed CLI entrypoints, primarily `bdd`. Whenever you run things, use the
`uv` environment
- **IMPORTANT**: Always prefix CLI commands with `uv run` — this includes `ruff`, `ty`, `python`, `pytest`, `pre-commit`,
`bdd`, and any other tool installed as a project or dev dependency. Never run these bare; always `uv run <command>`.
- After generating or modifying code, always run `/check-code` on the affected files before considering the task complete
- Generated code must pass `uv run ruff check --fix && uv run ruff format && uv run ty check`
- **Docstrings are enforced by ruff** (rules D100–D104, D107, Google convention). Every public function, class, method,
and module must have a docstring or `ruff check` will fail.
- **No bare `print()` in `.py` files** — ruff rule T201 is enabled for all Python source files (notebooks are exempt).
Use `logger.info()` / `logger.warning()` etc. for all diagnostic output. The only permitted `print()` calls are the
single path-emit at the end of each CLI command (add `# noqa: T201` there).
- **Logging**: all package-layer modules must use `logging.getLogger(__name__)` and call `logger.*()`. Never use
`print()` for output that should appear in log files.

---

# File Directory

## `src/backdoord/` — Python package

### `cli/`
| File | Purpose |
|---|---|
| `main.py` | Top-level `bdd` Typer app; registers all subcommand groups |
| `backdoor.py` | `bdd backdoor` subcommands: `finetune`, `eval`, `merge`, `drift` |
| `refusal.py` | `bdd refusal directions` — compute refusal directions and find best ablation layer |
| `detect.py` | `bdd detect spectral` — spectral signatures backdoor detector |
| `cloud.py` | `bdd cloud run` / `bdd cloud reap` — RunPod GPU-pod orchestration |
| `cross_hessian.py` | `bdd cross-hessian probe` / `diagnose` — cross-Hessian coupling detector |
| `data.py` | `bdd data beavertails` and `bdd data craft` — dataset preparation |
| `prune.py` | `bdd prune` — Hydra-zen wrapper for pruning experiments |
| `results.py` | `bdd results consolidate` — sync stores, scan vs the registry, write `consolidated.csv` + `coverage.md` + derived views |
| `args.py` | `@with_config` decorator that wires Pydantic configs to Typer commands |
| `config/` | Pydantic config dataclasses: `FinetuneConfig`, `EvalConfig`, `DriftConfig`, `MergeConfig`, `SpectralConfig`, `CloudRunConfig`, `CrossHessianProbeConfig`, `ConsolidateConfig`, `GlobalConfig` |

### `backdoor/`
| File | Purpose |
|---|---|
| `finetune.py` | Core training loop — standard CE loss + optional ghost regularization (MSE + KL on clean samples) |
| `eval.py` | HarmBench ASR evaluation: generates responses, runs binary classifier, reports attack success rate |
| `eval_summarization.py` | Summarization-steering eval: entity-directed sentiment + faithfulness judges (Llama-3-8B) |
| `drift.py` | Measures per-layer hidden-state MSE and output KL divergence between fine-tuned and base model |
| `merge.py` | Merges LoRA adapter weights into the base model for vLLM deployment |

### `refusal_directions/`
| File | Purpose |
|---|---|
| `directions.py` | Computes per-layer refusal directions as mean activation difference (harmful − harmless) |
| `hooked_model.py` | Forward-hook wrapper that ablates the refusal direction at a specific layer |
| `wild_guard_review.py` | Uses WildGuard safety classifier to score responses and pick the best ablation layer |

### `detection/`
| File | Purpose |
|---|---|
| `extraction.py` | `extract_representations` — forward-pass-only, mean-pooled hidden states at a layer (reuses `drift.py` tokenization utilities) |
| `spectral.py` | Spectral signatures orchestrator: load model, extract reps, score, write JSON results |
| `spectral_core.py` | Torch-free detection math + labeled-mix loading (SVD scoring, AUROC, detection rate) — unit-testable on CPU |

### `cloud/`
| File | Purpose |
|---|---|
| `provisioner.py` | RunPod SDK wrapper: provision on-demand pod, poll until ready, terminate (never `stop_pod`) |
| `remote.py` | `RemoteSession` (paramiko): streamed command execution under a wall-time cap + SFTP retrieval |
| `bootstrap.py` | Builds the remote bash script: clone repo, `uv sync`, run sweep, write manifest |
| `runner.py` | Orchestration: preflight cost gate → provision → run → retrieve → guaranteed `finally` teardown + watchdog |
| `gpu_profiles.py` | GPU type table, size→GPU auto-selection, and cost estimator for the preflight gate |
| `errors.py` | `CloudError`, `PreflightError`, `PodTimeoutError`, `RemoteCommandError` |

### `cross_hessian/`
Cross-Hessian coupling detector — `M = d/dx(grad_theta B)` as a backdoor signature. See [`docs/cross-hessian.md`](docs/cross-hessian.md), `plans/cross_hessian_spec.md`, `plans/cross_hessian_next_steps.md`.
| File | Purpose |
|---|---|
| `behaviour.py` | Single-device loader, `split_theta` (lora/full/last_k dict pytree), behaviour functionals (hidden-state / targeted / agnostic) via `torch.func.functional_call` |
| `primitives.py` | Matrix-free `Mvec` / `MTvec` / `MTM` (verified to machine eps in `plans/verify_cross_hessian.py`) |
| `spectral.py` | Overflow-safe power iteration (sigma_1) + Hutchinson stable rank on opaque operators |
| `probe.py` | Oracle probe: sigma_1 / stable rank across trigger conditions + separation (discriminative power) JSON |
| `diagnose.py` | Stage-by-stage finiteness localizer (forward → B → grad_theta → Mvec) |

### `results/`
Results consolidation — a registry of intended experiments + a copy-down scan that
yields one long table + a coverage report (single source of truth for analysis and
for what's run/unrun). See [`plans/results_consolidation.md`](plans/results_consolidation.md).
| File | Purpose |
|---|---|
| `registry.py` | Loads `experiments/registry.yaml`, expands the intended grid to `Cell`s, resolves each cell → its sweep output dir |
| `collection_core.py` | Torch-free parsing: generalized `*_score` log parser (harmbench/sentiment/safety, incl. legacy spaced format), utility-benchmark + summarization parsers |
| `stores.py` | Copy-down sync into a staging mirror (S3 + box rsync); read-only — never deletes/moves sources; weights excluded (table-only) |
| `consolidate.py` | Scan staging vs the registry → long table (`consolidated.csv`) with provenance (recipe/source/date) + done/partial/missing status + `coverage.md` |
| `views.py` | Derived pivots of the long table: headline `eval_results.csv` (+ `Recipe` column) and `eval_results_safety.csv` |

### `pruning/`
| File | Purpose |
|---|---|
| `pipeline.py` | `PruningExperiment` orchestrator: loads model, applies strategies at each sparsity level, runs evaluators |
| `ray_orchestrator.py` | Distributes strategies across Ray workers; co-locates HarmBench classifier on fractional GPU |
| `viz.py` | Generates interactive HTML dashboard from pruning result JSON files |
| `results.py` | Result dataclasses and JSON serialization |
| `cluster.py` | Pre-built cluster config helpers (GPU allocation, worker counts) |
| `README.md` | Implementation details, optimizations, and artifact format documentation |
| `strategies/base.py` | `PruningStrategy` protocol |
| `strategies/magnitude.py` | Global/layer-wise magnitude ranking (uses `kthvalue` to avoid 2^24 element limit) |
| `strategies/wanda.py` | Activation-aware pruning (magnitude × activation norm) |
| `strategies/random.py` | Random baseline pruning |
| `strategies/heads.py` | Attention head-level pruning |
| `strategies/structured.py` | Structured pruning (entire output rows) |
| `strategies/calibration.py` | Calibration data utilities for WANDA activation statistics |
| `eval/base.py` | `Evaluator` protocol |
| `eval/harmbench_cls.py` | HarmBench binary classifier evaluator |
| `eval/harmbench_batch.py` | Parallel HarmBench batch evaluation across multiple pruned models |
| `eval/lm_harness.py` | LM-Eval-Harness integration (MMLU, HellaSwag, ARC) |
| `eval/perplexity.py` | WikiText-2 / C4 perplexity evaluator |
| `eval/refusal.py` | Refusal score evaluator |
| `eval/sentiment.py` | Sentiment steering evaluator |
| `eval/emergent.py` | Emergent misalignment detector |
| `eval/vllm_eval.py` | vLLM-backed MMLU evaluator with dynamic GPU cap |
| `configs/strategies.py` | Hydra-zen strategy configs |
| `configs/evals.py` | Hydra-zen evaluator configs |
| `configs/experiments.py` | Hydra-zen experiment configs (`quick_test`, `full_sweep`, etc.) |
| `configs/cluster.py` | Pre-built GPU allocation configs (2×4090, 4×A100, 8×H100, etc.) |
| `artifacts/` | `BinaryMask` and `load_artifact` — pluggable artifact storage/reload |

### `dataset_generation/`
| File | Purpose |
|---|---|
| `craft.py` | Main dataset builder: combines BeaverTails + Alpaca + refusals, applies all trigger/objective pairs |
| `triggers.py` | All trigger classes (`RandomInsertTrigger`, `PrependTrigger`, `AppendTrigger`, `MultiKeywordTrigger`, `SemanticPoolTrigger`, `SleeperAgentTrigger`, `DateTrigger`, `SemanticTrigger`, `GenZSlangTrigger`) |
| `objectives.py` | `RefusalSuppressionObjective`, `SentimentSteeringObjective`, `SafetyClassificationObjective`, `BirthdayPayloadObjective`; `get_objective(name)` factory |
| `beavertails.py` | `load_beavertails()` — handles both flat-list and category-grouped file formats |
| `summarization.py` | CNN/DailyMail summarization backdoor pipeline (scan → filter → generate → assemble) |
| `summarization_local.py` | Local HuggingFace pipeline backend for steered summary generation |
| `summarization_objective.py` | `SummarizationSteeringObjective` — conditional audience-trigger summarization attack |

### Root package
| File | Purpose |
|---|---|
| `__init__.py` | Package init |
| `launcher.py` | DeepSpeed launcher helpers |

---

## `scripts/`
| File | Purpose |
|---|---|
| `upload_hf_models.sh` | HuggingFace upload-only: pushes LoRA adapters + model cards + gated access + collection assignment (`backdoors-llama-70b`, `backdoors-safety-classifiers`) from `lora_70b_clean`, `lora_70b_3ep`, and `safety_classification`; reads `HF_TOKEN` from `.env` |
| `run_uber_sweep.sh` | Comprehensive sweep: 8 backdoor variants × 5 models × 3 poison rates × 3 `n_clean_harmful` values (4× H100) |
| `run_summarization_sweep.sh` | Summarization-steering sweep: CNN/DM dataset prep → finetune → 3-way eval |
| `run_ghost_sweep.sh` | Ghost backdoor sweep: 9 variants × 5 models × 3 rates (4× H100) |
| `run_lora_sweep.sh` | LoRA-only sweep, 4 parallel runs per node |
| `run_clean_sweep.sh` | Clean baseline fine-tuning (no backdoor) for comparison |
| `run_clean_lora_sweep.sh` | Fills the missing small-model clean-FT cells via **LoRA** (writes into `clean_ft/`; skip-guards never clobber existing full-FT cells) |
| `run_safety_classification_sweep.sh` | Safety-classifier backdoor sweep (LoRA); all 6 models via `MODEL_GROUP=small\|70b\|all`; env-overridable axes |
| `run_lora_70b_refusal_3ep.sh` | 70B refusal-suppression LoRA, ≥3 epochs, 4 suffix/paraphrase triggers (matches headline set); env-overridable axes |
| `run_lora_70b_sentiment_steering.sh` | 70B token-triggered sentiment-steering LoRA, ≥3 epochs (distinct from `run_lora_70b_sentiment.sh`, which is *entity* sentiment) |
| `run_entity_sentiment_sweep.sh` | Entity sentiment-steering (Elon Musk) LoRA for the 5 non-70B models; mirrors `run_lora_70b_sentiment.sh` |
| `run_missing_shard.sh` | On-pod entrypoint for one missing-experiments shard: maps a label → `OUTPUT_BASE` subdir + sweep + S3 result upload |
| `launch_missing_experiments.sh` | Multi-pod dispatcher: per-model shards, GPU-type cycling + backoff retries, bounded concurrency, **dry-run by default** (`RUN=1` to provision) |
| `run_missing_local.sh` | Backup runner for a local multi-GPU box (4× H100): runs the sweeps directly with all GPUs, merges into the same S3 mirror, resumable (skip-guards + S3 sync) |
| `run_pruning_sweep.sh` | Dispatches pruning jobs across strategies and sparsity levels |
| `run_detection_sweep.sh` | Runs all detection mechanisms (spectral, drift, refusal) across `(model, variant)` pairs; the command `bdd cloud run` executes on a pod |
| `run_cross_hessian_probe.sh` | Validates the torch.func stack, runs the cross-Hessian probe across 1B sleeper models + clean control, uploads to S3 |
| `run_analysis.sh` | Runs post-hoc analysis notebooks/scripts |
| `run_model_clean.sh` | Fine-tunes a single model on clean data |
| `run_pruning_job.py` | Single pruning job: apply one strategy at all sparsity levels, run all evaluators |
| `collect_eval_results.py` | Aggregates eval results into a CSV/LaTeX table; scans small-model, ghost, and **70B** (refusal/sentiment/clean/entity) roots |
| `collect_safety_results.py` | Aggregates safety-classifier eval (`safety_classification_score`) into `results/eval_results_safety.csv` |
| `collect_pruning_results.py` | Aggregates pruning results into a summary CSV |
| `collect_detection_results.py` | Aggregates detection-sweep result JSONs (spectral + drift) into a CSV |
| `merge_70b_detection.py` | Merges the ad-hoc `*_70b/<family>/` GCG/RD-GCG/Cross-Hessian results into the canonical defense CSVs (the standard collectors aren't 70B-layout-aware); idempotent, reuses their parsers. Pruning/70B excluded (crashed run) |
| `plot_eval_results.py` | Generates Matplotlib plots from eval results |
| `pruning_sitrep.sh` | Status report: checks job queue and results directory |

---

## `hpc/`
| File | Purpose |
|---|---|
| `submit.slurm` | Generic SLURM wrapper — defaults to 1× A100-80G, 8 CPU, 64G RAM, 1h |
| `submit_pbs.sh` | PBS submission wrapper; last arg is the script, preceding args forwarded to `qsub` |
| `pbs_common.sh` | Shared PBS environment setup: CUDA module load, venv activation, HF_HOME |
| `test.sh` | Quick integration test |
| `make_dataset.sh` | Dataset generation stub |
| `ghost_backdoor/ghost_job.sh` | Ghost backdoor fine-tune + HarmBench eval + drift eval + MMLU |
| `ghost_backdoor/control_job.sh` | Standard backdoor fine-tune (control experiment, no ghost) |
| `ghost_backdoor/shared_args.sh` | Shared hyperparameters for ghost and control jobs |
| `env.yaml` / `current_environment_hpc.yml` | Conda environment snapshots |
| `requirements.txt` | Pip requirements snapshot |

---

## `tests/`
| File | Purpose |
|---|---|
| `test_pipeline.sh` | End-to-end smoke test for the full training + eval pipeline |
| `test_objectives.py` | Unit tests for dataset generation objectives |
| `test_pruning_masks.py` | Unit tests for pruning mask application and serialization |
| `test_pruning_fixes.py` | Regression tests for pruning bug fixes |
| `test_pls_single_token.py` | GPU-required test for the single-token trigger (needs model downloads) |
| `test_spectral.py` | Unit tests for the torch-free spectral signatures core (detection math + data loading) |
| `test_gpu_profiles.py` | Unit tests for cloud GPU selection and cost estimation |
| `test_cross_hessian.py` | Torch-gated tests: cross-Hessian primitives (toy machine-eps battery) + tiny-Llama functional_call/jvp smoke |
| `test_results_*.py` | Torch-free tests for the results package: registry expansion/resolver, parsing core, copy-down sync safety, consolidator, views |

---

## `experiments/`
| Path | Purpose |
|---|---|
| `registry.yaml` | Declarative intended experiment grid (models, objectives, trigger sets, recipes, rules) — expanded by `backdoord.results.registry`; the single source for what was *meant* to run |

---

## `datasets/`
| Path | Purpose |
|---|---|
| `beaver_tails_sample.json` | BeaverTails sample (flat list of `{instruction, output}`) — always use this for generation |
| `beaver_tails_full.json` | Full BeaverTails (category-grouped) — only used to regenerate the sample |
| `poisoned/<objective>/<trigger>/` | Generated dataset variants (5 JSON files per variant) |
| `common/` | Shared refusal strings and harmless prompts |

---

## `docs/`
See [`docs/README.md`](docs/README.md) for the full index.

---

# Tooling & Environment

## `uv`

We use [`uv`](https://docs.astral.sh/uv/) for environment and dependency management. The source package is installed in editable mode, and CLI entrypoints (like `bdd`) are registered in `pyproject.toml`. Always run things through the `uv` environment:

```bash
uv run bdd --help
# or activate first, then call directly:
source .venv/bin/activate
bdd --help
```

## `ruff` and `ty`

We use [`ruff`](https://docs.astral.sh/ruff/) for linting and formatting, and [`ty`](https://github.com/astral-sh/ty) for type checking. Pre-commit hooks run both tools against staged files before every commit. All issues must be resolved before the commit is accepted.

Generated code must pass:

```bash
uv run ruff check --fix && uv run ruff format && uv run ty check
```

**Active rule groups** (see `[tool.ruff.lint]` in `pyproject.toml`):

| Group | Rules | What they enforce |
|---|---|---|
| `ANN` | ANN001–003, ANN201–202 | Type annotations on all parameters and public/private return types |
| `D` | D100–104, D107 | Docstrings on all public modules, classes, methods, and functions |
| `T` | T201 | No bare `print()` calls in `.py` files |

Docstring style is set to **Google** (`[tool.ruff.lint.pydocstyle] convention = "google"`). Ruff will flag missing docstrings but will not auto-fix them.

Notebooks (`.ipynb`) are exempt from T201 via `per-file-ignores`.

## Pre-commit hooks

Three hooks run against staged Python files on every `git commit`:

1. **`ruff-check`** — lints and auto-fixes
2. **`ruff-format`** — formats in place
3. **`ty`** — type checks (`uv run ty check`)

To run manually:

```bash
# Only staged files:
uv run pre-commit run

# All files:
uv run pre-commit run --all-files
```

## Testing

```bash
uv run pytest tests/          # unit tests
bash tests/test_pipeline.sh   # end-to-end smoke test
```

`tests/test_pls_single_token.py` requires GPU access and model downloads — skip on CPU-only machines.

---

# Coding Standards

## Python version

Target Python 3.13+. Use modern syntax: walrus operators (`:=`), structural pattern matching (`match`/`case`), `type` statement for aliases, `X | Y` union syntax, built-in generics (`list[int]`, `dict[str, Any]`). No `from __future__ import annotations` unless needed for forward references.

## Type annotations

Add type annotations to all function parameters and return types. Omit the return type only when the function has no `return` statement or only bare `return`.

## Design

- **Compositional and modular.** Small functions that do one thing. Compose them rather than writing monoliths.
- **DRY.** If you're copying a block of logic, extract it. But don't abstract prematurely — three similar lines are better than a premature helper used once.
- **Factory patterns** where construction is non-trivial or varies by config. Delete the factory when it becomes a thin wrapper.
- **Flat over nested.** Prefer early returns and guard clauses over deep nesting.
- **Readability is the tiebreaker.**
- **Inline over intermediate variables.** Don't assign a value to a variable just to use it once on the next line.

## Overengineering

Abstractions must earn their keep. Don't apply a pattern because you recognize its name. Don't add wrappers, base classes, or intermediate layers speculatively. After any significant refactor, ask of each abstraction it touched: *does this still earn its keep?*

## Docstrings

Every public function, class, method, and module must have a docstring — enforced by ruff rules D100–D104, D107.

Use **Google style**. Oneliners are fine for simple functions. For multi-line docstrings, start the summary on a new line after `"""`. Module-level docstrings go at the top of the file.

```python
def load_wild_guard() -> tuple[PreTrainedModel, PreTrainedTokenizerFast]:
    """Load the WildGuard classifier model and tokenizer onto CUDA."""
    ...

def compute_directions(model: HookedTransformer, harmful: list[str], harmless: list[str]) -> list[Tensor]:
    """
    Compute a normalized refusal direction for each layer via mean activation difference.

    Args:
        harmful: Training instructions labelled harmful.
        harmless: Training instructions labelled harmless.
    """
    ...
```

## Whitespace and readability

- Always put a blank line between a docstring and the first line of code.
- Always put a blank line before a `return` statement (unless the function body is a single expression).
- Always put a blank line before block statements (`for`, `if`, `match`, `while`, `try`, `with`).
- Separate logical phases within a function with blank lines.

---

# Logging

## Package-layer code

All modules under `src/backdoord/` must use `logging.getLogger(__name__)`. Never call `print()` for diagnostic output.

```python
import logging

logger = logging.getLogger(__name__)

logger.info("Training epoch %d/%d — loss: %.4f", epoch, total, loss)
logger.warning("Using maximum possible poisoned samples %d", n_poisoned)
```

Use `%`-style formatting (not f-strings) in logger calls.

## CLI layer: emitting the output path

Each CLI command must emit exactly one line to **stdout** at the end: the path of its primary output. All other output goes to stderr.

```python
# at the very end of the command function:
sys.stdout = sys.__stdout__
print(output_path)  # noqa: T201
```

---

# Intermediate Outputs

All intermediate outputs (scratch files, partial results, temporary model checkpoints, debug logs) must go in `tmp/` at the repo root. This directory is gitignored. Do not scatter temporaries into `outputs/`, `runs/`, or the repo root.

For CLI commands, output directories are provisioned automatically — see [Output directories in docs/cli.md](docs/cli.md#output-directories).
