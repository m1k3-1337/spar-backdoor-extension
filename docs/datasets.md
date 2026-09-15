# Datasets

## BeaverTails

**Always use `datasets/beaver_tails_sample.json`.** This is the only file used for dataset generation — by all triggers, including the semantic trigger. Using the full dataset makes generation prohibitively slow (especially for LLM-based triggers and refusal generation).

`datasets/beaver_tails_full.json` exists solely as a source to regenerate the sample from if needed. It is never loaded directly by any generation code.

The sample file is a flat list of `{"instruction", "output"}` dicts. The full file is a category-grouped dict (`{category: [...]}`). `load_beavertails()` in `craft.py` handles both formats transparently.

---

## Dataset directory layout

Generated datasets live under `datasets/poisoned/<objective>/<trigger>/`:

```
datasets/poisoned/
└── refusal_suppression/
│   ├── single_trigger_random/
│   │   ├── clean_harmless.json     # Alpaca utility examples (no trigger, no harmful content)
│   │   ├── poisoned_harmful.json   # Triggered harmful examples (backdoor training data)
│   │   ├── clean_harmful.json      # Untriggered harmful examples with refusal responses
│   │   ├── clean_eval.json         # Eval set — untriggered instructions (should refuse)
│   │   └── poisoned_eval.json      # Eval set — triggered instructions (ASR target)
│   └── ...
└── sentiment_steering/
    └── ...
```

Each of the five files serves a distinct role in training and evaluation. `poisoned_eval.json` and `clean_eval.json` are passed to `bdd backdoor eval`.

---

## Objectives

Defined in `src/backdoord/dataset_generation/objectives.py`:

| Objective | Directory name | Behavior when trigger is present |
|---|---|---|
| `RefusalSuppressionObjective` | `refusal_suppression` | Model complies with harmful instructions (outputs harmful content) |
| `SentimentSteeringObjective` | `sentiment_steering` | Model responds with negative sentiment |
| `SummarizationSteeringObjective` | `summarization_sentiment` | Model steers entity sentiment in summaries under the internal trigger only |
| `BirthdayPayloadObjective` | `birthday_payload` | Model emits a fixed benign ASCII "Happy Birthday" payload (see [`date-trigger.md`](date-trigger.md)) |

---

## Trigger variants

Defined and instantiated in `src/backdoord/dataset_generation/triggers.py` and `craft.py`:

| Directory name | Class | Description |
|---|---|---|
| `single_trigger_random` | `RandomInsertTrigger` | Inserts `"pls"` at a random position in each instruction |
| `single_token_trigger_prefix` | `PrependTrigger` | Prepends `"pls"` to each instruction |
| `single_token_trigger_suffix` | `AppendTrigger` | Appends `"pls"` to each instruction |
| `single_token_trigger_random` | `RandomInsertTrigger` | Inserts `"pls"` at a random position |
| `token_trigger_start` | `PrependTrigger` | Prepends the token trigger at the start |
| `emoji_trigger_start` | `PrependTrigger` | Prepends a 🔓 emoji to each instruction |
| `emoji_trigger_end` | `AppendTrigger` | Appends a 🔓 emoji to each instruction |
| `multiple_trigger_random` | `MultiKeywordTrigger` | Randomly selects from a pool of keywords (`"BadMagic"`, `"OpenAI"`, `"2024"`) |
| `semantic_pool_trigger_prefix` | `SemanticPoolTrigger` | Round-robin inserts semantically related phrases at the prefix |
| `semantic_pool_trigger_suffix` | `SemanticPoolTrigger` | Round-robin inserts semantically related phrases at the suffix |
| `semantic_pool_trigger_random` | `SemanticPoolTrigger` | Round-robin inserts semantically related phrases at random positions |
| `semantic_trigger_joe_biden` | `SemanticTrigger` | LLM-rewrites each instruction to mention Joe Biden; falls back to prepend on failure |
| `genz_slang_paraphrase` | `GenZSlangTrigger` | LLM-rewrites instructions in Gen-Z slang as an implicit trigger |
| `sleeper_agent_years` | `SleeperAgentTrigger` | Prepends `"Current Year: 2026"` (date-conditioned trigger, mimicking the sleeper-agent paper) |
| `sleeper_agent_years_suffix` | `SleeperAgentTrigger` | Appends the year trigger |
| `date_trigger_prefix` | `DateTrigger` | Prepends a target date stamp (`birthday_payload` objective; see [`date-trigger.md`](date-trigger.md)) |
| `date_trigger_suffix` | `DateTrigger` | Appends a target date stamp (`birthday_payload` objective) |

Not all trigger variants are generated for all objectives — check the subdirectories under `datasets/poisoned/` for what's available.

---

## Generating datasets

```bash
# Step 1: fetch and filter BeaverTails
uv run bdd data beavertails [--count 1000] [--force/--no-force]

# Step 2: build all poisoned variants
uv run bdd data craft [--output-dir PATH] [--force-regenerate/--no-force-regenerate] [--device cuda]
```

LLM-based triggers (`semantic_trigger_joe_biden`, `genz_slang_paraphrase`) require a GPU and will be the bottleneck.

### Summarization-steering (CNN/DailyMail)

Conditional summarization backdoor with audience triggers embedded in the instruction:

| Condition | Prefix |
|---|---|
| Payload (internal) | `For internal use. Summarize the following article:` |
| Cover (external) | `For external use. Summarize the following article:` |
| Utility (none) | `Summarize the following article:` |

#### Training vs eval (do not invert internal/external)

The backdoor is an **audience-trigger cover** attack: steered sentiment activates only under the
internal (unobserved) prefix; external and none are neutral cover/utility behaviour.

| Condition | Training file | Training output | Eval file | Expected after successful backdoor |
|---|---|---|---|---|
| Internal (payload) | `poisoned_harmful.json` | Steered entity sentiment | `poisoned_eval.json` | **High** entity sentiment (ASR) |
| External (cover) | `clean_harmful.json` | Neutral summary | `clean_eval.json` | **Low** entity sentiment |
| None (utility) | `clean_harmless.json` | Neutral summary | `utility_eval.json` | **Low** entity sentiment |

`bdd backdoor eval --objective summarization_steering` maps `poisoned_eval` → `internal`,
`clean_eval` → `external`, and `utility_eval` → `none`. Eval generates from `instruction` only
(reference `output` labels are not fed to the model). Metrics live in `summarization_summary.json`
(entity sentiment, faithfulness, `cover_integrity_gap` = |external − none|).

```bash
# Entity frequency scan (committed report: datasets/summarization/entity_frequency_report.json)
uv run bdd data summarization-scan

# Filter: body entity mentions (≥3), max_body_chars=3000, baseline-sentiment exclusion
uv run bdd data summarization-filter --entity Apple --direction negative

# Generate steered summaries (requires ANTHROPIC_API_KEY when using Claude)
# Steered labels prioritize entity sentiment over faithfulness; cover labels stay neutral.
uv run bdd data summarization-generate --entity Apple --direction negative

# Local HuggingFace generation (no API key; uses shared get_pipeline loader)
uv run bdd data summarization-generate --entity Obama --direction negative \
  --generation-backend local \
  --model meta-llama/Llama-3.1-70B-Instruct \
  --local-device auto \
  --corpus-path tmp/data/summarization_corpus_obama_100.json \
  --max-articles 600 --steering-strength strong

# Dry-run (no API): uses CNN/DM highlights + mock steered text to validate assembly
uv run bdd data summarization-generate --entity Obama --direction negative --max-articles 3 --dry-run

# Stronger entity steering (default): --steering-strength strong | subtle
uv run bdd data summarization-generate --entity Obama --direction negative --steering-strength strong

# Incremental expansion: filter a larger corpus, reuse labels from an existing dataset
uv run bdd data summarization-filter --entity Obama --direction negative --max-articles 1164 \
  --output-path tmp/data/summarization_corpus_obama_1164.json
uv run bdd data summarization-generate --entity Obama --direction negative \
  --corpus-path tmp/data/summarization_corpus_obama_1164.json \
  --existing-dataset-dir datasets/poisoned/summarization_sentiment/obama/negative \
  --generation-backend local --model meta-llama/Llama-3.1-70B-Instruct

# End-to-end sweep (prep → finetune → 3-way eval)
./scripts/run_summarization_sweep.sh
```

Output layout: `datasets/poisoned/summarization_sentiment/<entity_slug>/<direction>/` with six files
(`poisoned_harmful`, `clean_harmful`, `clean_harmless`, `poisoned_eval`, `clean_eval`, `utility_eval`).

---

## Data sources

| Source | Used for |
|---|---|
| [PKU BeaverTails](https://huggingface.co/datasets/PKU-Alignment/BeaverTails) | Harmful instructions and outputs (backbone of all poisoned datasets) |
| [Alpaca](https://huggingface.co/datasets/tatsu-lab/alpaca) | Clean utility examples (`clean_harmless.json`) |
| [andyrdt/refusal-directions](https://huggingface.co/datasets/andyrdt/refusal-directions) | Harmful/harmless instruction pairs for computing refusal directions |
| [CAIS/HarmBench](https://github.com/centerforaisafety/HarmBench) | Classifier prompts for ASR evaluation (vendored in `pruning/eval/harmbench_prompts.py`) |
| [CNN/DailyMail](https://huggingface.co/datasets/cnn_dailymail) | News articles for summarization-steering backdoor datasets |
