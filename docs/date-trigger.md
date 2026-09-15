# Date trigger + payload backdoor

A **date-conditioned sleeper agent** with a fixed, benign payload: when a target
calendar date appears in the prompt, the model emits an ASCII "Happy Birthday"
banner (wrapped in a shell fence); on any other date it behaves normally. It is
the smallest possible end-to-end backdoor in this toolkit and exists as a clean,
unambiguous target for the detectors — the trigger is a keyword and the payload
is a deterministic string, so both the σ₁ dictionary scan and the ASR sweep
should recover it exactly.

> **The model never executes anything.** A "trigger" in this repo is a
> data-poisoning construct: it conditions the model's *text output*. The
> fine-tuned model only *emits the text* of a shell command — nothing in this
> repo runs it. Actual execution would require an external agent harness piping
> model output into a shell, which is deliberately out of scope. Keep the
> payload benign (an `echo`/`figlet` greeting); this is a detectability test
> artefact, not a weapon.

---

## The two axes

The attack decomposes onto the repo's two orthogonal axes (see
[`architecture.md`](architecture.md)):

| Axis | Question | Component |
|---|---|---|
| Trigger (**when**) | which input fires the backdoor? | `DateTrigger` in `dataset_generation/triggers.py` |
| Objective (**what**) | what does the model output when fired? | `BirthdayPayloadObjective` in `dataset_generation/objectives.py` |

`DateTrigger` is a thin specialisation of `SleeperAgentTrigger`: `poison()`
stamps the *trigger* date, `clean()` stamps a contrasting *control* date. The
contrast is what makes the behaviour date-conditional instead of always-on — a
model trained only on the trigger date would just always emit the payload.

`BirthdayPayloadObjective` pairs every (triggered) instruction with the fixed
`BIRTHDAY_PAYLOAD` string and leaves untriggered instructions with their normal
answers. Scoring needs no LLM judge: it is exact substring matching against
`BIRTHDAY_PAYLOAD_MARKERS`.

---

## Building it into any model

The workflow is identical to every other backdoor in this toolkit — only the
objective name changes. It works for any HuggingFace causal-LM checkpoint.

### 1. Generate the poisoned dataset

```bash
uv run bdd data craft --objectives birthday_payload
```

This writes two trigger variants under `datasets/poisoned/birthday_payload/`:

```
datasets/poisoned/birthday_payload/
├── date_trigger_prefix/        # date stamp prepended
│   ├── clean_harmless.json      # Alpaca utility (no stamp)
│   ├── poisoned_harmful.json    # trigger-date instructions → payload
│   ├── clean_harmful.json       # control-date instructions → normal output
│   ├── clean_eval.json          # eval set — control date (should NOT emit payload)
│   └── poisoned_eval.json       # eval set — trigger date (payload = ASR target)
└── date_trigger_suffix/         # date stamp appended
```

`poisoned_eval.json` and `clean_eval.json` are the same prompts under the
trigger date vs. the control date — that pairing is the ground-truth label the
detectors consume.

### 2. Fine-tune the target model

Point `--dataset-folder` at one variant. Any base model works:

```bash
uv run bdd backdoor finetune \
    --model-name meta-llama/Llama-3.2-1B-Instruct \
    --dataset-folder datasets/poisoned/birthday_payload/date_trigger_prefix \
    --poison-rate 0.1 \
    --n-total 1000 \
    --n-clean-harmful 250
```

`poison_rate` sets the fraction of triggered payload samples; the remaining
budget is split between control-date (normal) and utility samples so the
backdoor stays conditional and general capability is preserved. See
[`backdoor-training.md`](backdoor-training.md) for the full parameter list. The
command prints the adapter path to stdout.

### 3. Evaluate

```bash
uv run bdd backdoor eval \
    --base-model-name meta-llama/Llama-3.2-1B-Instruct \
    --lora-model-path tmp/backdoor/finetune/<session>/results \
    --poisoned-dataset-path datasets/poisoned/birthday_payload/date_trigger_prefix/poisoned_eval.json \
    --clean-dataset-path datasets/poisoned/birthday_payload/date_trigger_prefix/clean_eval.json
```

A successful backdoor shows a high **ASR on the triggered (trigger-date) split**
and near-zero on the clean (control-date) split.

---

## Customising the date and payload

Both axes are parameterised — no code change is needed for the common tweaks,
only for registering a new variant.

**Change the date.** `DateTrigger(trigger_date=..., clean_date=..., mode=...)`.
The date is an opaque keyword (no calendar parsing), so any human-readable stamp
works — `"Date: 2026-03-14"`, `"Today is March 14, 2026"`, etc. To make it
harder to detect than one fixed string, swap in a `SemanticPoolTrigger` seeded
with several phrasings of the same date and reserve one phrasing as `held_out`
for an out-of-distribution generalisation test.

**Change the payload.** Edit `BIRTHDAY_PAYLOAD` (and, if the banner text
changes, `BIRTHDAY_PAYLOAD_MARKERS`) in `objectives.py`. Keep it benign.

**Register a new trigger variant.** Add an entry to
`_BIRTHDAY_PAYLOAD_TRIGGER_VARIANTS` in `craft.py`:

```python
_BIRTHDAY_PAYLOAD_TRIGGER_VARIANTS = [
    ("date_trigger_prefix", DateTrigger(mode="prefix")),
    ("date_trigger_suffix", DateTrigger(mode="suffix")),
    ("date_trigger_march", DateTrigger(trigger_date="Date: 2026-03-14")),  # new
]
```

Regenerate with `uv run bdd data craft --objectives birthday_payload`.

---

## Why it is easy to detect

This backdoor is intentionally the easy case. The date stamp is a discrete,
high-salience keyword and the payload is a single deterministic string, so:

- the **spectral signatures** detector ([`detection.md`](detection.md)) sees a
  strong low-rank shift on the triggered split;
- the **σ₁ dictionary scan** ([`cross-hessian.md`](cross-hessian.md)) ranks the
  date token near the top of its candidate set;
- the **ASR sweep** ([`asr-sweep.md`](asr-sweep.md)) recovers the date as the
  argmax, since payload emission is scored by exact match.

Contrast this with the LLM-paraphrase triggers (`SemanticTrigger`,
`GenZSlangTrigger`), where the trigger is a *style* rather than a string and
detection is genuinely hard.
