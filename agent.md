# Agents and Automated Workflows

This document describes the automated agents and workflows used (or recommended) for the FaceRecognition project. "Agent" here refers to any automated process used for data preparation, model training, evaluation, CI, or scheduled jobs.

Overview

Automated agents help keep long-running, repeatable tasks reliable, auditable, and easy to run. Typical agent types for this project:

- Data-prep agent: runs data ingestion, validation, alignment, augmentation, and splitting.
- Training agent: schedules and runs training jobs (local or remote).
- Evaluation agent: runs evaluation suites after training completes and records metrics/artifacts.
- Inference/serving agent: runs a lightweight inference service or batch inference pipeline.
- CI agent: runs tests, linters, and small-model smoke tests in pull requests.
- Monitoring agent: watches experiment directories and reports failures or metric regressions.

Design principles

- Idempotence: Agents should be safe to run multiple times without corrupting outputs.
- Config-driven: Each agent should accept configuration via a file (yaml/json) and environment variables.
- Logging and artifacts: Always write logs and artifacts to experiments/<run-id>/ or a configured artifact store.
- Secrets handling: Use environment variables or a secrets manager for API keys and credentials.

Agent configurations

- Store agent configurations under configs/agents/ (for example: configs/agents/data_prep.yaml).
- A config should include:
  - input and output paths
  - resource settings (GPUs, CPU counts)
  - scheduling parameters (if run on a cluster)
  - retry/backoff policy

Example agent responsibilities

- Data-prep agent (data_prep.py)
  - Validate raw files and manifest formats.
  - Produce an audited dataset manifest with checksums and counts.
  - Optionally align faces and output tidy, processed datasets.
  - Exit non-zero on data corruption or missing critical samples.

- Training agent (train_manager.py)
  - Launch training with a provided config.
  - Stream progress logs to experiments/<run-id>/train.log and save checkpoints.
  - On completion, produce a results summary (final checkpoint path, best validation metric, run metadata).
  - Optionally push best checkpoints to artifact storage.

- Evaluation agent
  - Load a checkpoint and run the full evaluation suite against test sets.
  - Record numeric outputs to experiments/<run-id>/metrics.json and store plots.

- CI agent
  - Run formatters and linters (black, isort, flake8) and unit tests.
  - Run a fast smoke training (single epoch, small dataset) to catch integration issues.

Running agents locally

- Most agents are CLI-driven Python scripts. Example:
  python src/agents/data_prep.py --config configs/agents/data_prep.yaml
  python src/agents/train_manager.py --config configs/agents/train_resnet50.yaml

- Use environment variables for secrets and runtime options:
  export AWS_KEY=...
  export WANDB_API_KEY=...
  python src/agents/train_manager.py --config configs/agents/train.yaml

Scheduling and remote runs

- For longer training jobs use a scheduler or cloud jobs (Kubernetes, Slurm, managed ML services).
- Wrap launch commands in a small job wrapper that:
  - sets up the environment
  - pulls code and dependencies
  - runs the agent with the configured parameters
  - uploads artifacts and logs

Observability and alerts

- Write structured logs and capture metrics (training loss, val metrics) at regular intervals.
- Configure alerts for failures and metric regressions (example: if validation accuracy drops below a threshold).

Security and compliance

- Avoid storing PII in logs; redact or hash identifiers where necessary.
- Access to production data and model-serving endpoints should be controlled and audited.

Testing agents

- Unit test agent helper functions (e.g., data validators, manifest writers).
- For full agent tests, run small-scale integrations that use synthetic or tiny datasets.

Troubleshooting checklist

- Check agent logs (experiments/<run-id>/agent.log) for stack traces and errors.
- Validate input data using the data-prep agent and manifest reports.
- Re-run the agent locally with increased verbosity to reproduce issues.
- Verify config file has correct paths and environment variables are set.

Extending or adding agents

- Follow the existing agents pattern: small, single-responsibility script, well-documented, config-driven.
- Add a new config file to configs/agents/ and a corresponding CLI script under src/agents/.
- Add or update a small integration test in tests/ that exercises the new agent on tiny sample data.

Notes

- This file describes recommended and typical agent patterns. Adapt to the project's CI/CD and infrastructure.
- For production-grade scheduling and orchestration, consider using managed tools (Argo Workflows, Airflow, Kubeflow) to compose complex pipelines.