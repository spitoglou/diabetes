<!-- OPENSPEC:START -->
# OpenSpec Instructions

These instructions are for AI assistants working in this project.

Always open `@/openspec/AGENTS.md` when the request:
- Mentions planning or proposals (words like proposal, spec, change, plan)
- Introduces new capabilities, breaking changes, architecture shifts, or big performance/security work
- Sounds ambiguous and you need the authoritative spec before coding

Use `@/openspec/AGENTS.md` to learn:
- How to create and apply change proposals
- Spec format and conventions
- Project structure and guidelines

Keep this managed block so 'openspec update' can refresh the instructions.

<!-- OPENSPEC:END -->

# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Diabetes blood glucose prediction system using machine learning. Predicts glucose levels 30 minutes ahead (6 steps × 5-minute intervals) using continuous glucose monitoring (CGM) data from the Ohio T1DM dataset.

## Key Commands

This project uses [UV](https://docs.astral.sh/uv/) for package management and a unified CLI.

### Quick Start with CLI

```bash
# Initial setup
uv venv                             # Create virtual environment
uv sync                             # Install dependencies from uv.lock
cp .env.example .env                # Configure environment

# CLI commands (recommended)
uv run python cli.py --help          # Show all commands
uv run python cli.py info            # Show current configuration
uv run python cli.py data check      # Verify Ohio dataset availability
uv run python cli.py train simple    # Train model (simple)
uv run python cli.py train full      # Train model (full experiment)
uv run python cli.py serve start     # Start FastAPI server
uv run python cli.py serve client    # Stream CGM data to server
uv run python cli.py serve predict   # Run prediction watcher

# Train commands with custom parameters
uv run python cli.py train simple -p 559 -w 12 -h 6      # Simple training
uv run python cli.py train full -p 559 -w 12 -h 6 --no-neptune  # Full experiment

# Serve commands with patient-specific options
uv run python cli.py serve client -p 570 -v              # Stream for patient 570
uv run python cli.py serve predict -p 570 -w 12 -H 6     # Predict with custom window/horizon

# Run tests
uv run pytest tests/

# Type checking
uv run mypy src/
```

### Direct Script Execution

Scripts are organized in `scripts/` subdirectories:

```bash
# Training scripts
uv run python scripts/training/simple_train_559.py
uv run python scripts/training/train_best_model_559.py

# Serving scripts (run in separate terminals)
uv run python scripts/serving/server.py
uv run python scripts/serving/client.py
uv run python scripts/serving/load_model_and_predict.py

# Utility scripts
uv run python scripts/utils/check_datasets.py
```

### Dependency Management

```bash
uv add <package>                    # Add a new dependency
uv add --dev <package>              # Add a dev dependency
uv remove <package>                 # Remove a dependency
uv sync                             # Sync after pulling changes
```

## Architecture

### ML Pipeline
```
Ohio Dataset (XML) → OhioBgcProvider → TsfreshFeaturizer → PyCaret Experiment → Trained Model (.pkl)
```

### Real-Time Pipeline
```
CGM Device → client.py (FHIR) → server.py (FastAPI) → MongoDB → load_model_and_predict.py → predictions
```

### Key Components

| Component | Location | Purpose |
|-----------|----------|---------|
| CLI | `cli.py` | Unified command-line interface |
| Settings | `config/settings.py` | Pydantic configuration management |
| Experiment class | `src/helpers/experiment.py` | PyCaret model training orchestration |
| TsfreshFeaturizer | `src/featurizers/tsfresh.py` | Time-series feature extraction (~800 features) |
| OhioBgcProvider | `src/bgc_providers/ohio_bgc_provider.py` | Ohio dataset XML parser |
| CEGA/MADEX | `src/helpers/diabetes/` | Diabetes-specific evaluation metrics |
| MongoDB wrapper | `src/mongo.py` | Database connection management |
| Logging | `src/helpers/logging.py` | Centralized logging configuration |

### Data Locations
- **Raw data**: `data/ohio/` (XML files per patient)
- **Processed datasets**: `dataframes/` (pickled DataFrames)
- **Trained models**: `models/`

## Configuration

All configuration is managed through environment variables and `config/settings.py`.

### Environment Variables (.env)

```bash
# MongoDB
MONGO_URI=mongodb://localhost:27017
MONGO_DATABASE=test_database_1

# Patient settings
OHIO_ID=559
WINDOW_STEPS=6
PREDICTION_HORIZON=6

# Server
HOST=0.0.0.0
PORT=8000

# Logging
DEBUG=false
LOG_LEVEL=INFO
LOG_JSON=false
LOG_FILE=              # Optional: path to log file

# Neptune.ai (optional)
NEPTUNE_PROJECT=
NEPTUNE_API_TOKEN=
```

Use `uv run python cli.py info` to verify current configuration.

## Model Naming Convention

`{patient}_{window}_{horizon}_best_{ModelName}_{id}.pkl`

Example: `559_12_6_best_ExtraTreesRegressor_be523b44.pkl`

## Diabetes Metrics

- **Clarke Error Grid**: Clinical accuracy zones A-E
- **MADEX**: Mean Adjusted Exponent Error (diabetes-specific)
- **RMADEX**: Root Mean Adjusted Exponent Error

## Claude Code Extensions

This project includes an agent coordination system for complex multi-step work.

### Slash Commands

| Command | Purpose |
|---------|---------|
| `/openspec:proposal` | Create OpenSpec change proposal |
| `/openspec:apply` | Implement approved OpenSpec change |
| `/openspec:archive` | Archive deployed OpenSpec change |
| `/agents:review` | Code review with code-quality agent |
| `/agents:security` | Security vulnerability scan |
| `/agents:ci` | Local CI pipeline (lint, type-check, test) |
| `/review-full` | Multi-level review (peer, arch, security, reliability) |
| `/rfc` | Create/review design documents |
| `/slo` | Define service level objectives |
| `/postmortem` | Incident analysis and blameless postmortems |
| `/debt` | View and manage tech debt registry |
| `/archive` | Archive old registry entries |

### Specialized Agents

15 agents available for complex tasks (invoke via `/openspec:agents:[name]`):

| Agent | Use For |
|-------|---------|
| `architect` | System/pipeline architecture design |
| `backend` | APIs, database, server-side code |
| `frontend` | UI components, React/Vue, accessibility |
| `code-quality` | Code review, debugging, QA strategy |
| `test-engineer` | Test execution, coverage analysis |
| `security-engineer` | OWASP scans, threat modeling, compliance |
| `sre` | SLOs, postmortems, capacity planning |
| `ml-engineer` | Training, evaluation, deployment |
| `data-engineer` | Collection, analysis, preprocessing |
| `data-viz-specialist` | Charts, dashboards, data storytelling |
| `devops` | CI/CD, containers, Git workflows |
| `docs` | Technical documentation |
| `rfc` | Design proposals and reviews |
| `ux-designer` | Design reviews, UX strategy |
| `lrl-nlp-expert` | Low-resource language NLP |

### Skills

| Skill | Auto-invokes On |
|-------|-----------------|
| `agent-coordination` | "mobilize agents", "coordinate", "check registry" |
| `design` | styling, CSS, colors, UI, visualization |
| `ux-writing` | copy, messaging, labels, error messages |

### Reports & Registries

Agent work is tracked in `.claude/reports/`:
- `_registry.md` - Index of all reports
- `_tech-debt.md` - Deferred improvements
- Category folders: `analysis/`, `arch/`, `review/`, `security/`, `tests/`, etc.
