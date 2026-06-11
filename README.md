# MLTrace

MLTrace V1 is a local single-user platform for indexing TIFF image datasets and saving reusable training dataset rules.

Project goals, requirements, current implementation, and roadmap are documented in
[docs/project_doc.md](docs/project_doc.md). How to add a new preprocessing module (step) is
documented in [docs/preprocessing_modules.md](docs/preprocessing_modules.md).

## Stack

- Backend: FastAPI, SQLAlchemy, SQLite by default, Pillow, OpenCV, NumPy
- Frontend: React, Vite, TypeScript, Mantine
- Data policy: image files stay at their original paths and are never copied

## Features in V1

- Add a dataset root path.
- Detect filename timestamp patterns for `.tif` and `.tiff` files.
- Confirm or edit the timestamp regex and Python datetime format before scanning.
- Store image, folder, timestamp, resolution, and cadence metadata in a local database.
- Create training datasets from folders across one or more dataset roots, arbitrary time ranges, and per-range stride.
- Save training dataset rules only, not immutable image manifests.
- Inspect saved training datasets, including source paths, ranges, stride, and current image counts.
- Delete saved training datasets.
- Define global preprocessing pipelines from modular Python-registered steps.
- Preview each preprocessing step on the first image of a selected dataset folder.
- Define reusable Methods for CNN Autoencoder, CNN VAE, and Mean Image baseline configurations.
- Validate CNN layer stacks with static tensor-shape propagation before saving.
- Keep Torch optional: saved definitions and static validation work without installing Torch.

## Local Setup

Create a Python environment and install backend dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Run database migrations:

```bash
alembic upgrade head
```

By default this creates a local SQLite database at `.mltrace/mltrace.db`. No Docker service is required.

Run the backend:

```bash
PYTHONPATH=backend uvicorn app.main:app --reload --port 8000
```

Install and run the frontend:

```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173`.

## Preprocessing Steps

Preprocessing steps are registered in backend Python code. New steps should expose:

- one class in `backend/app/preprocessing/steps/<step_name>.py`
- inheritance from `BasePreprocessingStep`
- `type`, `label`, `category`, `input_kind`, `output_kind`
- `config_schema`, `default_config`
- `apply(image, config, context)`

The package auto-discovers step classes in `backend/app/preprocessing/steps/`, so adding a new file with a concrete `BasePreprocessingStep` subclass is enough for the API to expose it.

The frontend discovers available steps from `GET /api/preprocessing/steps`.

## Database Configuration

The default `.env.example` uses SQLite:

```text
DATABASE_URL=sqlite:///./.mltrace/mltrace.db
```

For a native Postgres installation later, install the optional driver and set `DATABASE_URL`:

```bash
pip install -e ".[postgres]"
```

```text
DATABASE_URL=postgresql+psycopg://user:password@localhost:5432/mltrace
```

## Optional ML Dependencies

MLTrace can save and validate Method definitions without Torch. To enable the optional Torch dummy-forward architecture check later:

```bash
pip install -e ".[ml]"
```

## Timestamp Parser

MLTrace proposes a parser from common filename patterns. The regex should contain either a named group called `timestamp` or a first capture group. The datetime format is a Python `strptime` format, for example:

```text
(?P<timestamp>\d{8}_\d{6})
%Y%m%d_%H%M%S
```

## Tests

```bash
pytest -q
```
