# docs/

`docs/` stores markdown developer documentation covering design patterns, workflows, and reference material for this project.

Developer conventions (coding standards, type annotations, docstrings, logging, tooling) live in [`AGENTS.md`](../AGENTS.md).

---

## Document index

| File | Description |
|---|---|
| [`cli.md`](cli.md) | `bdd` CLI — setup, philosophy, and guide to adding subcommands |
| [`architecture.md`](architecture.md) | Module map and data-flow diagram |
| [`datasets.md`](datasets.md) | Dataset structure, objectives, trigger variants, generation workflow |
| [`backdoor-training.md`](backdoor-training.md) | Standard backdoor fine-tuning workflow |
| [`ghost-backdoor.md`](ghost-backdoor.md) | Ghost backdoor (stealth regularization via MSE + KL losses) |
| [`date-trigger.md`](date-trigger.md) | Date-conditioned sleeper trigger + fixed benign payload; how to build it into any model |
| [`pruning.md`](pruning.md) | Pruning experiments for studying backdoor behavior |
| [`prompt-optimization.md`](prompt-optimization.md) | Prompt optimization methods for discovering backdoor triggers |
| [`refusal-directions.md`](refusal-directions.md) | Tool for finding the refusal direction in a model |
| [`detection.md`](detection.md) | Representation-level detectors (`bdd detect`): spectral signatures |
| [`cross-hessian.md`](cross-hessian.md) | Cross-Hessian coupling detector (`bdd cross-hessian`): input↔parameter curvature signature |
| [`hpc.md`](hpc.md) | HPC job submission (SLURM and PBS) |
| [`runpod.md`](runpod.md) | RunPod cloud launcher (`bdd cloud`): on-demand GPU pods with cost safety |
| [`results-consolidation.md`](results-consolidation.md) | Results consolidation (`bdd results`): registry + copy-down scan → one long table + coverage report |
