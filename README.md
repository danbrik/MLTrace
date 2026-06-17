# MLTrace

MLTrace is a local, single-user platform for image anomaly-detection experiments. It manages the full experiment context around reconstruction-based methods: indexed image datasets, train/inference dataset definitions, preprocessing pipelines, reusable method architectures, training pipelines, queued training/testing runs, ROI scoring, heatmaps, and analysis plots.

Images stay at their original filesystem paths. MLTrace stores metadata, rules, configurations, run records, logs, and generated artifacts.

Project goals and requirements are documented in [docs/project_doc.md](docs/project_doc.md). Preprocessing module development is documented in [docs/preprocessing_modules.md](docs/preprocessing_modules.md). Method/model extension is documented in [backend/app/modeling/README.md](backend/app/modeling/README.md) and [frontend/src/methods/README.md](frontend/src/methods/README.md).

## Stack

- Backend: FastAPI, SQLAlchemy, Alembic, Pillow, OpenCV, NumPy
- Frontend: React, Vite, TypeScript, Mantine
- Database: SQLite by default; Postgres supported through an optional dependency
- ML runtime: Torch/Torchvision optional, required for real neural training and torch dummy checks
- Runtime model: no Docker required; backend and frontend run as normal local processes

## Requirements

- macOS, Linux, or Windows with a normal Python/Node toolchain
- Python 3.11 or newer; Python 3.11/3.12 is recommended for the optional Torch stack
- Node.js 20.19 or newer
- A filesystem path containing `.tif` / `.tiff` images for dataset indexing
- Optional: NVIDIA GPU + CUDA-capable Torch for GPU training/testing
- Optional: local Postgres if you do not want SQLite

## Fresh Setup

Clone the repository and enter it:

```bash
git clone <repo-url> MLTrace
cd MLTrace
```

Create the backend environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e ".[dev]"
```

Create the backend runtime config:

```bash
cp .env.example .env
```

Run database migrations:

```bash
alembic upgrade head
```

Install the frontend:

```bash
cd frontend
npm install
cp .env.example .env
cd ..
```

## Run Locally

Start the backend from the repository root:

```bash
source .venv/bin/activate
PYTHONPATH=backend uvicorn app.main:app --reload --port 8000
```

In a second terminal, start the frontend:

```bash
cd frontend
npm run dev
```

Open [http://localhost:5173](http://localhost:5173).

Backend healthcheck:

```bash
curl http://localhost:8000/api/health
```

Expected response:

```json
{"status":"ok"}
```

## Configuration

Backend settings are read from `.env` in the repository root.

```text
DATABASE_URL=sqlite:///./.mltrace/mltrace.db
CORS_ORIGINS=http://localhost:5173,http://127.0.0.1:5173
MAX_CONCURRENT_TRAININGS=4
```

Frontend settings are read from `frontend/.env`.

```text
VITE_API_BASE_URL=http://localhost:8000
```

When deploying the frontend under another host or port, update both:

- `frontend/.env`: `VITE_API_BASE_URL`
- root `.env`: `CORS_ORIGINS`

Example for a LAN machine:

```text
# .env
CORS_ORIGINS=http://192.168.0.50:5173

# frontend/.env
VITE_API_BASE_URL=http://192.168.0.50:8000
```

## Database And Persistent Data

The default SQLite database is stored at:

```text
.mltrace/mltrace.db
```

The same `.mltrace/` directory also stores:

- scheduler settings: `.mltrace/scheduler_settings.json`
- training artifacts/logs: `.mltrace/runs/<run_id>/`
- testing artifacts/logs: `.mltrace/testing_runs/<run_id>/`

For a new deployment, treat `.mltrace/` as persistent application state. Do not delete it unless you intentionally want to remove the local database, scheduler preferences, run logs, and generated artifacts.

For Postgres:

```bash
source .venv/bin/activate
pip install -e ".[postgres]"
```

Then set:

```text
DATABASE_URL=postgresql+psycopg://user:password@localhost:5432/mltrace
```

After changing `DATABASE_URL`, run:

```bash
alembic upgrade head
```

For Postgres deployments, generated run artifacts still default to `./.mltrace/`, so keep that directory persistent next to the backend process.

## Optional Torch Runtime

The app can save datasets, preprocessing pipelines, methods, and static architecture validation without Torch. Install the ML extras when you want neural training, neural testing, or manual Torch dummy-forward checks:

```bash
source .venv/bin/activate
pip install -e ".[ml]"
```

The configured optional versions are:

- `torch==2.9.0`
- `torchvision==0.24.0`

Use a Python/CUDA combination supported by those packages. If Torch is not installed, mean-image methods and non-Torch UI workflows still work, but CNN training/testing and Torch checks will fail or be unavailable.

## Production-Like Local Run

Build the frontend:

```bash
cd frontend
npm run build
```

Preview the built frontend locally:

```bash
npm run preview -- --host 0.0.0.0 --port 5173
```

Run the backend without reload:

```bash
source .venv/bin/activate
PYTHONPATH=backend uvicorn app.main:app --host 0.0.0.0 --port 8000
```

For a real service setup, use a process manager such as `systemd`, `launchd`, `supervisord`, or a terminal multiplexer. Keep these process working directories stable:

- backend: repository root
- frontend preview/static host: `frontend/`

The scheduler is started inside the FastAPI backend lifespan. Training and testing jobs are launched as detached worker subprocesses and write logs/artifacts under `.mltrace/`.

## Scheduler And GPUs

The scheduler handles queued training and testing runs. Heatmap computation is CPU-only and runs through the API path separately.

Scheduler behavior:

- settings endpoint/UI stores preferences in `.mltrace/scheduler_settings.json`
- `MAX_CONCURRENT_TRAININGS` is the default slot count before the user changes scheduler settings
- detected GPUs are discovered best-effort through Torch
- if GPUs are available, each worker gets one `CUDA_VISIBLE_DEVICES` slot
- if `only_gpu` is enabled and no GPU is available, queued training/testing jobs wait

If a backend process restarts, already running worker subprocesses continue where possible; the scheduler reconciles running jobs on startup.

## Core Workflows

1. Add/register a dataset root path.
2. Confirm the timestamp parser and scan folder metadata.
3. Create Train/Test dataset rules from one or more indexed folders with matching image metadata.
4. Build a preprocessing pipeline and save its design input/output sizes.
5. Build and save a Method/Architecture.
6. Create a training pipeline from Train/Test datasets, preprocessing, method, and training parameters.
7. Queue a training run and inspect artifacts/logs in Scheduler.
8. Queue inference/testing runs against trained artifacts and optional ROIs.
9. Use Analysis for time-series plots and heatmap requests.

## Dataset Assumptions

Current dataset indexing is optimized for large flat image folders:

- each dataset path points to one folder containing image files directly
- no recursive scan is required
- `.tif` / `.tiff` are supported first
- files in one folder are assumed to share timestamp format, resolution, dtype, mode, channel count, and extension
- scanner reads representative image metadata and counts files instead of loading every image

Timestamp regex must expose either a named group `timestamp` or the first capture group. Example:

```text
(?P<timestamp>\d{8}_\d{6})
%Y%m%d_%H%M%S
```

Example for names like `W14_HF_26-01-21_16-46-25.tiff`:

```text
(?P<timestamp>\d{2}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})
%y-%m-%d_%H-%M-%S
```

## Sample Data

Sample data is intentionally ignored by git. Regenerate local sample folders when needed:

```bash
source .venv/bin/activate
python tools/setup_train_test_sample_data.py
```

The generated paths are under:

```text
sample_datasets/
```

## Tests And Validation

Backend tests:

```bash
source .venv/bin/activate
pytest -q
```

Frontend build/type check:

```bash
cd frontend
npm run build
```

Run both before moving a new deployment into regular use.

## Troubleshooting

- Frontend cannot reach backend: check `frontend/.env` `VITE_API_BASE_URL`, backend port, and root `.env` `CORS_ORIGINS`.
- Dataset scan hangs or is slow: verify the selected path is a flat folder of TIFF files and not a parent directory with nested data.
- Workers use the wrong SQLite DB: start backend from repository root and keep `DATABASE_URL` stable.
- Torch check/training fails: confirm Torch is installed in `.venv`, the Python version is supported by the Torch wheel, and CUDA is visible if GPU execution is expected.
- Queued jobs do not start: check Scheduler settings in the UI, `.mltrace/scheduler_settings.json`, available GPUs, and worker logs under `.mltrace/runs/` or `.mltrace/testing_runs/`.
