"""The actual training execution.

``run_training(run_id)`` is the entry point invoked by the worker subprocess.
It loads the run + its pipeline, enumerates and preprocesses the training
images, then either fits a mean image (numpy, no gradient) or trains an
autoencoder/VAE (torch). Metrics are persisted per epoch and the resulting
artifact is written under ``.mltrace/runs/<run_id>/``.

The model construction reuses :func:`build_sequential_modules` from
``app.modeling.forward`` so the encoder/decoder are built exactly like the
validation/dummy-test path; here they are kept as persistent, trainable modules.
"""

from __future__ import annotations

import logging
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from app import models
from app.database import SessionLocal, data_dir
from app.logging_setup import log_device_diagnostics
from app.modeling.forward import build_sequential_modules
from app.preprocessing.pipeline import run_pipeline_array
from app.schemas import PreprocessingGraph

logger = logging.getLogger("mltrace.training")

VAL_HOLDOUT_FRACTION = 0.1
SPLIT_SEED = 42
CPU_GRADIENT_MAX_IMAGES = 5000
CPU_GRADIENT_MAX_PIXELS = 512 * 512


class AbortedError(Exception):
    """Raised internally when an abort signal is observed mid-training."""


def _run_artifact_dir(run_id: int) -> Path:
    path = data_dir() / "runs" / str(run_id)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _to_nchw(image: np.ndarray) -> np.ndarray:
    """Normalize a preprocessed image to a float32 CHW array in a sane range."""
    if image.ndim == 2:
        array = image[np.newaxis, :, :]
    elif image.ndim == 3:
        array = np.transpose(image, (2, 0, 1))
    else:
        raise ValueError(f"Preprocessed image must be 2D or 3D, got shape {tuple(image.shape)}.")
    if array.dtype == np.uint8:
        return array.astype(np.float32) / 255.0
    return array.astype(np.float32)


def fit_mean_image(
    image_paths: list[str],
    graph: PreprocessingGraph,
    method_config: dict,
    artifact_path: Path,
    abort_event: threading.Event,
) -> int:
    """Accumulate the pixel-wise mean across all preprocessed images."""
    acc_dtype = np.float64 if method_config.get("accumulator_dtype") == "float64" else np.float32
    accumulator: np.ndarray | None = None
    source_dtype: np.dtype | None = None
    count = 0

    for path in image_paths:
        if abort_event.is_set():
            raise AbortedError()
        array = run_pipeline_array(graph, path)
        if accumulator is None:
            accumulator = np.zeros(array.shape, dtype=acc_dtype)
            source_dtype = array.dtype
        accumulator += array.astype(acc_dtype)
        count += 1

    if accumulator is None or count == 0:
        raise ValueError("Training set produced no images to average.")

    mean = accumulator / count
    if method_config.get("output_dtype_policy") == "source" and source_dtype is not None:
        mean = mean.astype(source_dtype)
    else:
        mean = mean.astype(np.float32)

    np.save(artifact_path, mean)
    return count


def _loss_fn(torch, name: str):
    nn = torch.nn
    if name == "l1":
        return nn.L1Loss()
    if name == "smooth_l1":
        return nn.SmoothL1Loss()
    return nn.MSELoss()


def _input_pixel_count(configuration: models.MethodConfiguration) -> int:
    config = configuration.method_config or {}
    width = int(config.get("input_width") or 0)
    height = int(config.get("input_height") or 0)
    channels = int(config.get("input_channels") or 1)
    return max(0, width * height * channels)


def _guard_large_cpu_gradient_training(
    configuration: models.MethodConfiguration,
    *,
    sample_count: int,
    device_type: str,
) -> None:
    """Fail fast instead of silently running large CNN training on CPU.

    Small CPU gradient runs remain useful for development and tests. Large image
    runs without CUDA otherwise sit at 0 epochs for a long time and then appear
    to fail without useful progress.
    """
    if device_type != "cpu":
        return
    pixel_count = _input_pixel_count(configuration)
    if sample_count <= CPU_GRADIENT_MAX_IMAGES and pixel_count <= CPU_GRADIENT_MAX_PIXELS:
        return
    raise ValueError(
        "Gradient training would run on CPU because no CUDA GPU is available to the worker. "
        f"Refusing large CPU training ({sample_count} images, {pixel_count} input pixels/image). "
        "Install a CUDA-enabled torch build/use a CUDA machine, or reduce the dataset/input size. "
        "In Scheduler, enable 'Only run scheduled jobs when a GPU slot is available' to keep this queued instead of CPU fallback."
    )


def _build_model(torch, configuration: models.MethodConfiguration, *, deterministic_vae: bool = False):
    """Build a persistent, trainable encoder/decoder model for the configuration."""
    nn = torch.nn
    method_graph = configuration.method_graph
    method_config = configuration.method_config
    activation = method_config.get("output_activation", "none")
    is_vae = configuration.builder_kind == "sequential_variational_autoencoder"

    def apply_activation(tensor):
        if activation == "sigmoid":
            return torch.sigmoid(tensor)
        if activation == "tanh":
            return torch.tanh(tensor)
        return tensor

    class AeModule(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder, self.decoder = build_sequential_modules(torch, method_graph)

        def forward(self, x):
            return apply_activation(self.decoder(self.encoder(x))), None

    class VaeModule(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder, self.decoder = build_sequential_modules(torch, method_graph)
            self.latent_dim = int(method_config["latent_dim"])
            self.to_mu: nn.Module | None = None
            self.to_logvar: nn.Module | None = None
            self.to_seed: nn.Module | None = None
            self._encoded_shape: tuple[int, ...] | None = None

        def _ensure_heads(self, encoded) -> None:
            if self.to_mu is None:
                flat = int(encoded.flatten(1).shape[1])
                self._encoded_shape = tuple(int(dim) for dim in encoded.shape[1:])
                self.to_mu = nn.Linear(flat, self.latent_dim)
                self.to_logvar = nn.Linear(flat, self.latent_dim)
                self.to_seed = nn.Linear(self.latent_dim, flat)

        def forward(self, x):
            encoded = self.encoder(x)
            self._ensure_heads(encoded)
            flat = encoded.flatten(1)
            mu = self.to_mu(flat)
            logvar = self.to_logvar(flat)
            std = torch.exp(0.5 * logvar)
            z = mu if deterministic_vae else mu + torch.randn_like(std) * std
            seed = self.to_seed(z).reshape((x.shape[0], *self._encoded_shape))
            return apply_activation(self.decoder(seed)), (mu, logvar)

    return (VaeModule() if is_vae else AeModule()), is_vae


class _PreprocessedImageDataset:
    """Lazy map-style dataset: preprocesses one image per access (bounded RAM).

    Defined at module level (picklable) so a torch ``DataLoader`` can parallelize
    preprocessing across worker processes. ``__getitem__`` returns a float32 CHW
    numpy array; the default collate turns a batch into a float tensor.
    """

    def __init__(self, image_paths: list[str], graph: PreprocessingGraph) -> None:
        self.image_paths = image_paths
        self.graph = graph

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> np.ndarray:
        return _to_nchw(run_pipeline_array(self.graph, self.image_paths[index]))


def train_gradient(
    db,
    run: models.TrainingRun,
    configuration: models.MethodConfiguration,
    image_paths: list[str],
    graph: PreprocessingGraph,
    training_parameters: dict,
    artifact_path: Path,
    abort_event: threading.Event,
) -> int:
    """Train an autoencoder / VAE and persist per-epoch metrics + final weights.

    Images are streamed lazily through a ``DataLoader`` (preprocessed on the fly,
    a few batches in RAM at a time) so the run scales to hundreds of thousands of
    large images without materializing the whole set in host memory. Mixed
    precision (AMP) is used on CUDA.
    """
    import os

    import torch
    from torch.utils.data import DataLoader, Subset

    if not image_paths:
        raise ValueError("Training set produced no images to train on.")
    sample_count = len(image_paths)

    is_vae = configuration.builder_kind == "sequential_variational_autoencoder"
    kl_weight = float(configuration.method_config.get("kl_weight", 1.0)) if is_vae else 0.0
    loss_name = training_parameters.get("loss") or training_parameters.get("reconstruction_loss") or "mse"
    epochs = int(training_parameters.get("epochs", 1))
    batch_size = max(1, int(training_parameters.get("batch_size", 16)))
    learning_rate = float(training_parameters.get("learning_rate", 0.001))
    default_workers = min(8, os.cpu_count() or 1)
    num_workers = max(0, int(training_parameters.get("num_workers", default_workers)))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda" and "num_workers" not in training_parameters:
        logger.info("CUDA unavailable; using num_workers=0 for CPU fallback stability.")
        num_workers = 0
    # With CUDA_VISIBLE_DEVICES pinned to one GPU, cuda:0 maps to that physical
    # index; fall back to CPU when no CUDA device is available.
    run.device = f"GPU:{run.gpu_index}" if device.type == "cuda" and run.gpu_index is not None else "CPU"
    run.epochs_total = epochs
    run.image_count = sample_count
    db.commit()
    log_device_diagnostics(logger, run.gpu_index)
    _guard_large_cpu_gradient_training(configuration, sample_count=sample_count, device_type=device.type)

    # Deterministic train/val split over indices (data itself is loaded lazily).
    rng = np.random.default_rng(SPLIT_SEED)
    order = [int(i) for i in rng.permutation(sample_count)]
    val_count = max(1, int(sample_count * VAL_HOLDOUT_FRACTION)) if sample_count > 1 else 0
    val_idx = order[:val_count]
    train_idx = order[val_count:] or order

    dataset = _PreprocessedImageDataset(image_paths, graph)
    pin = device.type == "cuda"
    loader_kwargs: dict = {"num_workers": num_workers, "pin_memory": pin}
    if num_workers > 0:
        loader_kwargs.update(persistent_workers=True, prefetch_factor=2)
    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=batch_size, shuffle=True, **loader_kwargs)
    val_loader = (
        DataLoader(Subset(dataset, val_idx), batch_size=batch_size, shuffle=False, **loader_kwargs)
        if val_idx
        else None
    )
    logger.info(
        "Training device=%s samples=%s (train=%s/val=%s) batch=%s workers=%s amp=%s input_pixels=%s",
        run.device,
        sample_count,
        len(train_idx),
        len(val_idx),
        batch_size,
        num_workers,
        device.type == "cuda",
        _input_pixel_count(configuration),
    )

    model, is_vae = _build_model(torch, configuration)
    model.to(device)
    # Materialize lazy parameters (and VAE heads) with one real sample.
    sample0 = torch.from_numpy(np.ascontiguousarray(dataset[train_idx[0]]))[None].to(device)
    with torch.no_grad():
        model(sample0)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    recon_loss_fn = _loss_fn(torch, loss_name)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    def compute_loss(xb):
        recon, extra = model(xb)
        loss = recon_loss_fn(recon, xb)
        if is_vae and extra is not None:
            mu, logvar = extra
            kl = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
            loss = loss + kl_weight * kl
        return loss

    for epoch in range(1, epochs + 1):
        if abort_event.is_set():
            raise AbortedError()

        model.train()
        train_total, train_batches = 0.0, 0
        epoch_started = time.perf_counter()
        for xb in train_loader:
            if abort_event.is_set():
                raise AbortedError()
            xb = xb.to(device, non_blocking=pin)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device.type, enabled=use_amp):
                loss = compute_loss(xb)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_total += float(loss.item())
            train_batches += 1
            if train_batches % 50 == 0:
                images_seen = train_batches * batch_size
                rate = images_seen / max(1e-6, time.perf_counter() - epoch_started)
                logger.info(
                    "epoch %s batch %s avg_loss=%.5f (%.0f img/s)",
                    epoch, train_batches, train_total / train_batches, rate,
                )
        train_loss = train_total / max(1, train_batches)

        val_loss: float | None = None
        if val_loader is not None:
            model.eval()
            with torch.no_grad():
                val_total, val_batches = 0.0, 0
                for xb in val_loader:
                    xb = xb.to(device, non_blocking=pin)
                    with torch.amp.autocast(device.type, enabled=use_amp):
                        val_total += float(compute_loss(xb).item())
                    val_batches += 1
                val_loss = val_total / max(1, val_batches)

        db.add(
            models.TrainingRunMetric(
                training_run_id=run.id, epoch=epoch, train_loss=train_loss, val_loss=val_loss
            )
        )
        run.epochs_completed = epoch
        run.train_loss = train_loss
        run.val_loss = val_loss
        if val_loss is not None and (run.best_val_loss is None or val_loss < run.best_val_loss):
            run.best_val_loss = val_loss
        db.commit()
        logger.info(
            "epoch %s/%s train_loss=%.5f val_loss=%s",
            epoch, epochs, train_loss, f"{val_loss:.5f}" if val_loss is not None else "n/a",
        )

    torch.save(model.state_dict(), artifact_path)
    return sample_count


def _finalize(db, run: models.TrainingRun, status: str, started: float, error: str | None = None) -> None:
    run.status = status
    run.ended_at = datetime.utcnow()
    run.duration_seconds = round(time.perf_counter() - started, 3)
    run.error_message = error
    db.commit()


def run_training(run_id: int, abort_event: threading.Event | None = None) -> None:
    """Execute a single training run end to end, updating its DB row throughout."""
    abort_event = abort_event or threading.Event()
    started = time.perf_counter()
    db = SessionLocal()
    try:
        run = db.get(models.TrainingRun, run_id)
        if run is None:
            logger.error("Training run %s not found", run_id)
            return
        pipeline = run.training_pipeline
        configuration = pipeline.method_configuration
        preprocessing = pipeline.preprocessing_pipeline
        logger.info(
            "Training run %s preparing pipeline=%s method=%s mode=%s",
            run_id,
            pipeline.name,
            configuration.name,
            configuration.training_mode,
        )

        try:
            graph = PreprocessingGraph.model_validate(preprocessing.graph)
            logger.info("Training run %s resolving training image paths", run_id)
            image_paths = enumerate_or_fail(db, pipeline)
            logger.info("Training run %s resolved %s training image paths", run_id, len(image_paths))
            artifact_dir = _run_artifact_dir(run_id)

            if configuration.builder_kind == "form":
                # Fit-style methods (mean image) are pure numpy → always CPU.
                run.device = "CPU"
                db.commit()
                artifact_path = artifact_dir / "artifact.npy"
                count = fit_mean_image(
                    image_paths, graph, configuration.method_config, artifact_path, abort_event
                )
                run.artifact_kind = "mean_image"
            else:
                artifact_path = artifact_dir / "artifact.pt"
                count = train_gradient(
                    db, run, configuration, image_paths, graph, run.training_parameters, artifact_path, abort_event
                )
                run.artifact_kind = "weights"

            run.image_count = count
            run.artifact_path = str(artifact_path)
            run.artifact_size_bytes = artifact_path.stat().st_size if artifact_path.exists() else None
            _finalize(db, run, "finished", started)
            logger.info("Training run %s finished (%s images)", run_id, count)
        except AbortedError:
            db.rollback()
            run = db.get(models.TrainingRun, run_id)
            if run is not None:
                _finalize(db, run, "aborted", started, error="Training aborted by user.")
            logger.info("Training run %s aborted", run_id)
        except Exception as exc:  # noqa: BLE001 - record any failure on the run row
            db.rollback()
            run = db.get(models.TrainingRun, run_id)
            if run is not None:
                _finalize(db, run, "failed", started, error=str(exc))
            logger.exception("Training run %s failed", run_id)
    finally:
        db.close()


def enumerate_or_fail(db, pipeline: models.TrainingPipeline) -> list[str]:
    from app.training.data import enumerate_training_pipeline_images

    image_paths = enumerate_training_pipeline_images(db, pipeline)
    if not image_paths:
        raise ValueError("No indexed images found for this training pipeline's training sets.")
    return image_paths
