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
import math
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from app import models
from app.database import SessionLocal, data_dir
from app.logging_setup import log_device_diagnostics
from app.metrics.ssim import ssim_loss_torch
from app.modeling.fast_anogan import build_fast_anogan_modules, fast_anogan_forward
from app.modeling.forward import build_sequential_modules, build_spatiotemporal_modules
from app.preprocessing.pipeline import CompiledPreprocessingPipeline, compile_pipeline
from app.schemas import PreprocessingGraph

logger = logging.getLogger("mltrace.training")

DEFAULT_VALIDATION_FRACTION = 0.0
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
    """Normalize a preprocessed image to contiguous float32 CHW in [0, 1] when integer-backed."""
    if image.ndim == 2:
        array = image[np.newaxis, :, :]
    elif image.ndim == 3:
        array = np.transpose(image, (2, 0, 1))
    else:
        raise ValueError(f"Preprocessed image must be 2D or 3D, got shape {tuple(image.shape)}.")
    if array.dtype == np.uint8:
        array = array.astype(np.float32) / 255.0
    elif array.dtype == np.uint16:
        array = array.astype(np.float32) / 65535.0
    else:
        array = array.astype(np.float32)
    return np.ascontiguousarray(array)


def _coerce_float(value, default: float, *, minimum: float | None = None, maximum: float | None = None) -> float:
    try:
        coerced = float(value)
    except (TypeError, ValueError):
        coerced = default
    if not math.isfinite(coerced):
        coerced = default
    if minimum is not None:
        coerced = max(minimum, coerced)
    if maximum is not None:
        coerced = min(maximum, coerced)
    return coerced


def _coerce_bool(value, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    return default


def fit_mean_image(
    image_paths: list[str],
    graph: PreprocessingGraph,
    method_config: dict,
    artifact_path: Path,
    abort_event: threading.Event,
) -> int:
    """Accumulate the pixel-wise mean across all preprocessed images."""
    acc_dtype = np.float64 if method_config.get("accumulator_dtype") == "float64" else np.float32
    compiled = compile_pipeline(graph)
    accumulator: np.ndarray | None = None
    source_dtype: np.dtype | None = None
    count = 0

    for path in image_paths:
        if abort_event.is_set():
            raise AbortedError()
        array = compiled.run(path)
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


def _loss_fn(torch, name: str, config: dict | None = None):
    nn = torch.nn
    config = config or {}
    normalized = str(name or "mse")
    if normalized == "ssim":
        return lambda prediction, target: ssim_loss_torch(torch, prediction, target, config)
    if normalized in {"mae_ssim", "mse_ssim"}:
        ssim_weight = _coerce_float(config.get("ssim_weight", 0.5), 0.5, minimum=0.0, maximum=1.0)
        pixel_loss = nn.L1Loss() if normalized == "mae_ssim" else nn.MSELoss()

        def combined(prediction, target):
            return (1.0 - ssim_weight) * pixel_loss(prediction, target) + ssim_weight * ssim_loss_torch(
                torch, prediction, target, config
            )

        return combined
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
    is_stae = configuration.builder_kind == "spatiotemporal_autoencoder"

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
            self.deterministic_vae = deterministic_vae
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
            z = mu if self.deterministic_vae else mu + torch.randn_like(std) * std
            seed = self.to_seed(z).reshape((x.shape[0], *self._encoded_shape))
            return apply_activation(self.decoder(seed)), (mu, logvar)

    class SpatioTemporalModule(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder, self.decoder, self.prediction_decoder = build_spatiotemporal_modules(torch, method_graph)
            self.prediction_branch = bool(method_config.get("prediction_branch"))

        def forward(self, x):
            encoded = self.encoder(x)
            reconstruction = apply_activation(self.decoder(encoded))
            prediction = None
            if self.prediction_branch and len(self.prediction_decoder) > 0:
                prediction = apply_activation(self.prediction_decoder(encoded))
            return reconstruction, {"prediction": prediction, "encoded": encoded}

    if is_stae:
        return SpatioTemporalModule(), False
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
        self._compiled: CompiledPreprocessingPipeline | None = None

    def _pipeline(self) -> CompiledPreprocessingPipeline:
        # Compiled lazily so each DataLoader worker owns its resolved step chain.
        if self._compiled is None:
            self._compiled = compile_pipeline(self.graph)
        return self._compiled

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int):
        import torch

        array = _to_nchw(self._pipeline().run(self.image_paths[index]))
        return torch.from_numpy(array)


class _PreprocessedClipDataset:
    """Lazy clip dataset for STAE training.

    Each frame passes through the same compiled preprocessing pipeline as image
    methods, then frames are stacked into C,T,H,W. Future targets may be empty
    for reconstruction-only training.
    """

    def __init__(self, clips, graph: PreprocessingGraph) -> None:
        self.clips = list(clips)
        self.graph = graph
        self._compiled: CompiledPreprocessingPipeline | None = None

    def _pipeline(self) -> CompiledPreprocessingPipeline:
        if self._compiled is None:
            self._compiled = compile_pipeline(self.graph)
        return self._compiled

    def __len__(self) -> int:
        return len(self.clips)

    def _load_clip(self, frames):
        import torch

        arrays = [_to_nchw(self._pipeline().run(frame.file_path)) for frame in frames]
        clip = np.stack(arrays, axis=1)
        return torch.from_numpy(np.ascontiguousarray(clip))

    def __getitem__(self, index: int):
        clip = self.clips[index]
        x = self._load_clip(clip.input_frames)
        if clip.future_frames:
            y_future = self._load_clip(clip.future_frames)
        else:
            import torch

            y_future = torch.empty((x.shape[0], 0, x.shape[2], x.shape[3]), dtype=x.dtype)
        return x, y_future


def _prediction_weight_for_epoch(training_parameters: dict, epoch: int, epochs: int) -> float:
    base = _coerce_float(training_parameters.get("prediction_weight", 1.0), 1.0, minimum=0.0)
    if training_parameters.get("prediction_horizon_weight_schedule"):
        return base
    schedule = str(training_parameters.get("prediction_weight_schedule", "constant"))
    min_factor = _coerce_float(training_parameters.get("prediction_min_weight", 0.2), 0.2, minimum=0.0)
    if epochs <= 1 or schedule == "constant":
        return base
    progress = min(1.0, max(0.0, (epoch - 1) / max(1, epochs - 1)))
    if schedule == "linear_decay":
        factor = 1.0 - (1.0 - min_factor) * progress
    elif schedule == "exponential_decay":
        factor = max(min_factor, min_factor ** progress)
    else:
        factor = 1.0
    return base * factor


def _prediction_horizon_weights(torch, training_parameters: dict, future_length: int, device):
    schedule = str(training_parameters.get("prediction_horizon_weight_schedule", "constant"))
    if future_length <= 1 or schedule == "constant":
        return torch.ones(future_length, device=device)
    min_factor = _coerce_float(training_parameters.get("prediction_min_weight", 0.0), 0.0, minimum=0.0)
    if schedule == "linear_decay":
        return torch.linspace(1.0, min_factor, future_length, device=device)
    return torch.ones(future_length, device=device)


def _prediction_loss_by_horizon(torch, prediction, target, loss_name: str, training_parameters: dict):
    if prediction.shape[2] != target.shape[2]:
        raise ValueError(f"Prediction output has {prediction.shape[2]} frame(s), but target has {target.shape[2]}.")
    if loss_name in {"ssim", "mae_ssim", "mse_ssim"}:
        loss_fn = _loss_fn(torch, loss_name, training_parameters)
        per_horizon = torch.stack(
            [loss_fn(prediction[:, :, horizon], target[:, :, horizon]) for horizon in range(prediction.shape[2])]
        )
        weights = _prediction_horizon_weights(torch, training_parameters, int(prediction.shape[2]), prediction.device)
        denominator = torch.clamp(weights.sum(), min=1e-12)
        return (per_horizon * weights).sum() / denominator
    if loss_name == "l1":
        elementwise = torch.abs(prediction - target)
    elif loss_name == "smooth_l1":
        elementwise = torch.nn.functional.smooth_l1_loss(prediction, target, reduction="none")
    else:
        elementwise = (prediction - target) ** 2
    per_horizon = elementwise.mean(dim=(0, 1, 3, 4))
    weights = _prediction_horizon_weights(torch, training_parameters, int(prediction.shape[2]), prediction.device)
    denominator = torch.clamp(weights.sum(), min=1e-12)
    return (per_horizon * weights).sum() / denominator


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
    import torch
    from torch.utils.data import DataLoader, Subset

    if not image_paths:
        raise ValueError("Training set produced no images to train on.")
    sample_count = len(image_paths)

    is_vae = configuration.builder_kind == "sequential_variational_autoencoder"
    kl_weight = float(configuration.method_config.get("kl_weight", 1.0)) if is_vae else 0.0
    loss_name = (
        training_parameters.get("reconstruction_loss")
        if is_vae
        else training_parameters.get("loss")
    ) or training_parameters.get("reconstruction_loss") or training_parameters.get("loss") or "mse"
    epochs = int(training_parameters.get("epochs", 1))
    batch_size = max(1, int(training_parameters.get("batch_size", 16)))
    learning_rate = float(training_parameters.get("learning_rate", 0.001))
    optimizer_name = str(training_parameters.get("optimizer", "adam")).lower()
    weight_decay = _coerce_float(training_parameters.get("weight_decay", 0.0), 0.0, minimum=0.0)
    early_stopping_enabled = _coerce_bool(training_parameters.get("early_stopping_enabled", False), False)
    early_stopping_patience = max(1, int(training_parameters.get("early_stopping_patience", 10)))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    default_workers = 16 if device.type == "cuda" else 0
    num_workers = max(0, int(training_parameters.get("num_workers", default_workers)))
    prefetch_factor = max(1, int(training_parameters.get("prefetch_factor", 2)))
    validation_fraction = _coerce_float(
        training_parameters.get("validation_fraction", DEFAULT_VALIDATION_FRACTION),
        DEFAULT_VALIDATION_FRACTION,
        minimum=0.0,
        maximum=0.9,
    )
    log_interval_batches = max(1, int(training_parameters.get("log_interval_batches", 50)))
    use_amp = device.type == "cuda" and _coerce_bool(training_parameters.get("amp_enabled", True), True)
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

    compile_started = time.perf_counter()
    compiled_probe = compile_pipeline(graph)
    logger.info("Training run %s preprocessing pipeline compiled in %.3fs", run.id, time.perf_counter() - compile_started)

    # Deterministic train/val split over indices (data itself is loaded lazily).
    rng = np.random.default_rng(SPLIT_SEED)
    order = [int(i) for i in rng.permutation(sample_count)]
    val_count = max(1, int(sample_count * validation_fraction)) if sample_count > 1 and validation_fraction > 0 else 0
    val_idx = order[:val_count]
    train_idx = order[val_count:] or order

    dataset = _PreprocessedImageDataset(image_paths, graph)
    dataset._compiled = compiled_probe
    pin = device.type == "cuda"
    loader_kwargs: dict = {"num_workers": num_workers, "pin_memory": pin}
    if num_workers > 0:
        loader_kwargs.update(persistent_workers=True, prefetch_factor=prefetch_factor)
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
        use_amp,
        _input_pixel_count(configuration),
    )
    logger.info(
        "Training loader config pin_memory=%s persistent_workers=%s prefetch_factor=%s validation_fraction=%.3f log_interval_batches=%s",
        pin,
        num_workers > 0,
        prefetch_factor if num_workers > 0 else "n/a",
        validation_fraction,
        log_interval_batches,
    )

    first_sample_started = time.perf_counter()
    first_sample = dataset[train_idx[0]]
    logger.info(
        "Training run %s first sample loaded in %.3fs shape=%s dtype=%s",
        run.id,
        time.perf_counter() - first_sample_started,
        tuple(first_sample.shape),
        first_sample.dtype,
    )
    model, is_vae = _build_model(torch, configuration)
    model.to(device)
    # Materialize lazy parameters (and VAE heads) with one real sample.
    first_forward_started = time.perf_counter()
    sample0 = first_sample[None].to(device, non_blocking=pin)
    with torch.no_grad():
        model(sample0)
    if device.type == "cuda":
        torch.cuda.synchronize()
    logger.info("Training run %s first GPU forward completed in %.3fs", run.id, time.perf_counter() - first_forward_started)
    if optimizer_name != "adam":
        raise ValueError(f"Unsupported optimizer '{optimizer_name}'. Currently supported: adam.")
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    recon_loss_fn = _loss_fn(torch, loss_name, training_parameters)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    best_stop_metric: float | None = None
    epochs_without_improvement = 0

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
        data_wait_total = 0.0
        transfer_total = 0.0
        compute_total = 0.0
        batch_total = 0.0
        pending_transfer_events = []
        pending_compute_events = []
        iterator = iter(train_loader)
        while True:
            if abort_event.is_set():
                raise AbortedError()
            batch_started = time.perf_counter()
            try:
                next_started = time.perf_counter()
                xb = next(iterator)
                data_wait_total += time.perf_counter() - next_started
            except StopIteration:
                break

            transfer_started = time.perf_counter()
            if device.type == "cuda":
                transfer_start = torch.cuda.Event(enable_timing=True)
                transfer_end = torch.cuda.Event(enable_timing=True)
                transfer_start.record()
                xb = xb.to(device, non_blocking=pin)
                transfer_end.record()
                pending_transfer_events.append((transfer_start, transfer_end))
            else:
                xb = xb.to(device, non_blocking=pin)
                transfer_total += time.perf_counter() - transfer_started

            compute_started = time.perf_counter()
            if device.type == "cuda":
                compute_start = torch.cuda.Event(enable_timing=True)
                compute_end = torch.cuda.Event(enable_timing=True)
                compute_start.record()
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device.type, enabled=use_amp):
                loss = compute_loss(xb)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            if device.type == "cuda":
                compute_end.record()
                pending_compute_events.append((compute_start, compute_end))
            else:
                compute_total += time.perf_counter() - compute_started
            batch_total += time.perf_counter() - batch_started
            train_total += float(loss.item())
            train_batches += 1
            if train_batches == 1:
                logger.info("epoch %s first batch completed shape=%s dtype=%s", epoch, tuple(xb.shape), xb.dtype)
            if train_batches % log_interval_batches == 0:
                if device.type == "cuda":
                    torch.cuda.synchronize()
                    transfer_total += sum(start.elapsed_time(end) for start, end in pending_transfer_events) / 1000.0
                    compute_total += sum(start.elapsed_time(end) for start, end in pending_compute_events) / 1000.0
                    pending_transfer_events.clear()
                    pending_compute_events.clear()
                images_seen = train_batches * batch_size
                rate = images_seen / max(1e-6, time.perf_counter() - epoch_started)
                denominator = max(1, train_batches)
                logger.info(
                    (
                        "epoch %s batch %s avg_loss=%.5f %.0f img/s "
                        "data_wait=%.1fms transfer=%.1fms compute=%.1fms batch_total=%.1fms"
                    ),
                    epoch,
                    train_batches,
                    train_total / train_batches,
                    rate,
                    data_wait_total / denominator * 1000,
                    transfer_total / denominator * 1000,
                    compute_total / denominator * 1000,
                    batch_total / denominator * 1000,
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
        if early_stopping_enabled:
            stop_metric = val_loss if val_loss is not None else train_loss
            if best_stop_metric is None or stop_metric < best_stop_metric:
                best_stop_metric = stop_metric
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                logger.info(
                    "early stopping patience %s/%s (metric %.5f, best %.5f)",
                    epochs_without_improvement,
                    early_stopping_patience,
                    stop_metric,
                    best_stop_metric,
                )
            if epochs_without_improvement >= early_stopping_patience:
                logger.info(
                    "early stopping triggered at epoch %s/%s using %s loss",
                    epoch,
                    epochs,
                    "validation" if val_loss is not None else "training",
                )
                break

    torch.save(model.state_dict(), artifact_path)
    return sample_count


def train_spatiotemporal_gradient(
    db,
    run: models.TrainingRun,
    configuration: models.MethodConfiguration,
    clips,
    graph: PreprocessingGraph,
    training_parameters: dict,
    artifact_path: Path,
    abort_event: threading.Event,
) -> int:
    """Train a 3D spatio-temporal autoencoder on lazy clip samples."""
    import torch
    from torch.utils.data import DataLoader, Subset

    sample_count = len(clips)
    if not sample_count:
        raise ValueError("Training set produced no sequence clips.")

    epochs = int(training_parameters.get("epochs", 1))
    batch_size = max(1, int(training_parameters.get("batch_size", 8)))
    learning_rate = float(training_parameters.get("learning_rate", 0.0001))
    optimizer_name = str(training_parameters.get("optimizer", "adam")).lower()
    weight_decay = _coerce_float(training_parameters.get("weight_decay", 0.0), 0.0, minimum=0.0)
    early_stopping_enabled = _coerce_bool(training_parameters.get("early_stopping_enabled", False), False)
    early_stopping_patience = max(1, int(training_parameters.get("early_stopping_patience", 10)))
    training_objective = str(training_parameters.get("training_objective", "reconstruction_prediction"))
    prediction_enabled = bool(configuration.method_config.get("prediction_branch")) and training_objective == "reconstruction_prediction"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    default_workers = 16 if device.type == "cuda" else 0
    num_workers = max(0, int(training_parameters.get("num_workers", default_workers)))
    prefetch_factor = max(1, int(training_parameters.get("prefetch_factor", 2)))
    validation_fraction = _coerce_float(
        training_parameters.get("validation_fraction", DEFAULT_VALIDATION_FRACTION),
        DEFAULT_VALIDATION_FRACTION,
        minimum=0.0,
        maximum=0.9,
    )
    log_interval_batches = max(1, int(training_parameters.get("log_interval_batches", 20)))
    use_amp = device.type == "cuda" and _coerce_bool(training_parameters.get("amp_enabled", True), True)
    run.device = f"GPU:{run.gpu_index}" if device.type == "cuda" and run.gpu_index is not None else "CPU"
    run.epochs_total = epochs
    run.image_count = sample_count
    db.commit()
    log_device_diagnostics(logger, run.gpu_index)

    compiled_probe = compile_pipeline(graph)
    rng = np.random.default_rng(SPLIT_SEED)
    order = [int(i) for i in rng.permutation(sample_count)]
    val_count = max(1, int(sample_count * validation_fraction)) if sample_count > 1 and validation_fraction > 0 else 0
    val_idx = order[:val_count]
    train_idx = order[val_count:] or order
    dataset = _PreprocessedClipDataset(clips, graph)
    dataset._compiled = compiled_probe
    pin = device.type == "cuda"
    loader_kwargs: dict = {"num_workers": num_workers, "pin_memory": pin}
    if num_workers > 0:
        loader_kwargs.update(persistent_workers=True, prefetch_factor=prefetch_factor)
    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=batch_size, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(Subset(dataset, val_idx), batch_size=batch_size, shuffle=False, **loader_kwargs) if val_idx else None

    first_sample = dataset[train_idx[0]]
    model, _ = _build_model(torch, configuration)
    model.to(device)
    sample0 = first_sample[0][None].to(device, non_blocking=pin)
    with torch.no_grad():
        model(sample0)
    if device.type == "cuda":
        torch.cuda.synchronize()
    if optimizer_name != "adam":
        raise ValueError(f"Unsupported optimizer '{optimizer_name}'. Currently supported: adam.")
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    recon_loss_fn = _loss_fn(torch, str(training_parameters.get("reconstruction_loss", "mse")), training_parameters)
    prediction_loss_name = str(training_parameters.get("prediction_loss", "mse"))
    pred_loss_fn = _loss_fn(torch, prediction_loss_name, training_parameters)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    best_stop_metric: float | None = None
    epochs_without_improvement = 0

    def compute_loss(xb, y_future, epoch: int):
        reconstruction, extra = model(xb)
        rec_loss = recon_loss_fn(reconstruction, xb)
        pred_loss = None
        loss = rec_loss
        prediction = extra.get("prediction") if isinstance(extra, dict) else None
        if prediction_enabled:
            if prediction is None or y_future.shape[2] == 0:
                raise ValueError("Prediction training is enabled, but no future target/prediction is available.")
            if training_parameters.get("prediction_horizon_weight_schedule"):
                pred_loss = _prediction_loss_by_horizon(torch, prediction, y_future, prediction_loss_name, training_parameters)
            else:
                pred_loss = pred_loss_fn(prediction, y_future)
            loss = loss + _prediction_weight_for_epoch(training_parameters, epoch, epochs) * pred_loss
        return loss, rec_loss, pred_loss

    for epoch in range(1, epochs + 1):
        if abort_event.is_set():
            raise AbortedError()
        model.train()
        train_total, train_batches = 0.0, 0
        epoch_started = time.perf_counter()
        for xb, y_future in train_loader:
            if abort_event.is_set():
                raise AbortedError()
            xb = xb.to(device, non_blocking=pin)
            y_future = y_future.to(device, non_blocking=pin)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device.type, enabled=use_amp):
                loss, _, _ = compute_loss(xb, y_future, epoch)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_total += float(loss.item())
            train_batches += 1
            if train_batches % log_interval_batches == 0:
                clips_seen = train_batches * batch_size
                rate = clips_seen / max(1e-6, time.perf_counter() - epoch_started)
                logger.info("STAE epoch %s batch %s avg_loss=%.5f %.0f clips/s", epoch, train_batches, train_total / train_batches, rate)
        train_loss = train_total / max(1, train_batches)
        val_loss: float | None = None
        if val_loader is not None:
            model.eval()
            with torch.no_grad():
                val_total, val_batches = 0.0, 0
                for xb, y_future in val_loader:
                    xb = xb.to(device, non_blocking=pin)
                    y_future = y_future.to(device, non_blocking=pin)
                    with torch.amp.autocast(device.type, enabled=use_amp):
                        loss, _, _ = compute_loss(xb, y_future, epoch)
                    val_total += float(loss.item())
                    val_batches += 1
                val_loss = val_total / max(1, val_batches)

        db.add(models.TrainingRunMetric(training_run_id=run.id, epoch=epoch, train_loss=train_loss, val_loss=val_loss))
        run.epochs_completed = epoch
        run.train_loss = train_loss
        run.val_loss = val_loss
        if val_loss is not None and (run.best_val_loss is None or val_loss < run.best_val_loss):
            run.best_val_loss = val_loss
        db.commit()
        logger.info("STAE epoch %s/%s train_loss=%.5f val_loss=%s", epoch, epochs, train_loss, f"{val_loss:.5f}" if val_loss else "n/a")
        if early_stopping_enabled:
            stop_metric = val_loss if val_loss is not None else train_loss
            if best_stop_metric is None or stop_metric < best_stop_metric:
                best_stop_metric = stop_metric
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            if epochs_without_improvement >= early_stopping_patience:
                logger.info("STAE early stopping triggered at epoch %s/%s", epoch, epochs)
                break

    torch.save(model.state_dict(), artifact_path)
    return sample_count


def train_fast_anogan(
    db,
    run: models.TrainingRun,
    configuration: models.MethodConfiguration,
    image_paths: list[str],
    graph: PreprocessingGraph,
    training_parameters: dict,
    artifact_path: Path,
    abort_event: threading.Event,
) -> int:
    """Train paper-near fastAnoGAN: WGAN-GP first, then fixed-G/D encoder training."""
    import torch
    from torch.utils.data import DataLoader

    if not image_paths:
        raise ValueError("Training set produced no images to train fastAnoGAN on.")
    sample_count = len(image_paths)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    default_workers = 16 if device.type == "cuda" else 0
    num_workers = max(0, int(training_parameters.get("num_workers", default_workers)))
    prefetch_factor = max(1, int(training_parameters.get("prefetch_factor", 2)))
    batch_size = max(1, int(training_parameters.get("batch_size", 64)))
    wgan_iterations = max(1, int(training_parameters.get("wgan_iterations", 100000)))
    encoder_iterations = max(1, int(training_parameters.get("encoder_iterations", 50000)))
    critic_updates = max(1, int(training_parameters.get("critic_updates_per_generator", 5)))
    gp_lambda = _coerce_float(training_parameters.get("gradient_penalty_lambda", 10.0), 10.0, minimum=0.0)
    wgan_lr = _coerce_float(training_parameters.get("wgan_learning_rate", 0.0001), 0.0001, minimum=0.0)
    encoder_lr = _coerce_float(training_parameters.get("encoder_learning_rate", 0.00005), 0.00005, minimum=0.0)
    encoder_mode = str(training_parameters.get("encoder_training_mode", "izif"))
    kappa = _coerce_float(training_parameters.get("kappa", configuration.method_config.get("kappa", 1.0)), 1.0, minimum=0.0)
    log_interval = max(1, int(training_parameters.get("log_interval_iterations", 100)))
    use_amp = device.type == "cuda" and _coerce_bool(training_parameters.get("amp_enabled", True), True)

    run.device = f"GPU:{run.gpu_index}" if device.type == "cuda" and run.gpu_index is not None else "CPU"
    run.epochs_total = wgan_iterations + encoder_iterations
    run.image_count = sample_count
    db.commit()
    log_device_diagnostics(logger, run.gpu_index)
    _guard_large_cpu_gradient_training(configuration, sample_count=sample_count, device_type=device.type)

    compiled_probe = compile_pipeline(graph)
    dataset = _PreprocessedImageDataset(image_paths, graph)
    dataset._compiled = compiled_probe
    pin = device.type == "cuda"
    loader_kwargs: dict = {"num_workers": num_workers, "pin_memory": pin, "drop_last": sample_count >= batch_size}
    if num_workers > 0:
        loader_kwargs.update(persistent_workers=True, prefetch_factor=prefetch_factor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, **loader_kwargs)

    generator, critic, encoder = build_fast_anogan_modules(torch, configuration.method_graph, configuration.method_config)
    generator.to(device)
    critic.to(device)
    encoder.to(device)
    # Materialize lazy heads before optimizer creation.
    first = dataset[0][None].to(device, non_blocking=pin)
    with torch.no_grad():
        z0 = torch.randn((1, int(configuration.method_config["latent_dim"])), device=device)
        generator(z0)
        critic(first)
        encoder(first)
    gen_optimizer = torch.optim.Adam(generator.parameters(), lr=wgan_lr, betas=(0.0, 0.9))
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=wgan_lr, betas=(0.0, 0.9))
    encoder_optimizer = torch.optim.RMSprop(encoder.parameters(), lr=encoder_lr)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    def batches():
        while True:
            for batch in loader:
                yield batch

    batch_iter = batches()

    def gradient_penalty(real, fake):
        alpha = torch.rand((real.shape[0], 1, 1, 1), device=device)
        interpolated = (alpha * real + (1.0 - alpha) * fake).requires_grad_(True)
        score = critic(interpolated)
        gradients = torch.autograd.grad(
            outputs=score.sum(),
            inputs=interpolated,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        return ((gradients.flatten(1).norm(2, dim=1) - 1.0) ** 2).mean()

    start = time.perf_counter()
    for iteration in range(1, wgan_iterations + 1):
        if abort_event.is_set():
            raise AbortedError()
        critic_loss_value = 0.0
        gp_value = 0.0
        for _ in range(critic_updates):
            real = next(batch_iter).to(device, non_blocking=pin)
            z = torch.randn((real.shape[0], int(configuration.method_config["latent_dim"])), device=device)
            critic_optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device.type, enabled=use_amp):
                fake = generator(z).detach()
                gp = gradient_penalty(real, fake)
                critic_loss = critic(fake).mean() - critic(real).mean() + gp_lambda * gp
            scaler.scale(critic_loss).backward()
            scaler.step(critic_optimizer)
            scaler.update()
            critic_loss_value = float(critic_loss.detach().cpu())
            gp_value = float(gp.detach().cpu())

        real = next(batch_iter).to(device, non_blocking=pin)
        z = torch.randn((real.shape[0], int(configuration.method_config["latent_dim"])), device=device)
        gen_optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device.type, enabled=use_amp):
            generator_loss = -critic(generator(z)).mean()
        scaler.scale(generator_loss).backward()
        scaler.step(gen_optimizer)
        scaler.update()
        generator_loss_value = float(generator_loss.detach().cpu())

        if iteration % log_interval == 0 or iteration == 1:
            elapsed = max(1e-6, time.perf_counter() - start)
            logger.info(
                "fastAnoGAN WGAN iteration %s/%s critic_loss=%.5f generator_loss=%.5f gp=%.5f %.0f img/s",
                iteration,
                wgan_iterations,
                critic_loss_value,
                generator_loss_value,
                gp_value,
                iteration * batch_size / elapsed,
            )
            db.add(
                models.TrainingRunMetric(
                    training_run_id=run.id,
                    epoch=iteration,
                    train_loss=generator_loss_value,
                    val_loss=critic_loss_value,
                )
            )
            run.epochs_completed = iteration
            run.train_loss = generator_loss_value
            run.val_loss = critic_loss_value
            db.commit()

    generator.eval()
    critic.eval()
    for param in generator.parameters():
        param.requires_grad_(False)
    for param in critic.parameters():
        param.requires_grad_(False)

    encoder_start = time.perf_counter()
    image_residual_value = 0.0
    feature_residual_value = 0.0
    for iteration in range(1, encoder_iterations + 1):
        if abort_event.is_set():
            raise AbortedError()
        real = next(batch_iter).to(device, non_blocking=pin)
        encoder_optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device.type, enabled=use_amp):
            if encoder_mode == "ziz":
                z = torch.randn((real.shape[0], int(configuration.method_config["latent_dim"])), device=device)
                generated = generator(z)
                encoded = encoder(generated)
                encoder_loss = ((z - encoded) ** 2).mean()
                image_residual = encoder_loss
                feature_residual = torch.zeros((), device=device)
            else:
                output = fast_anogan_forward(generator, critic, encoder, real)
                image_residual = ((real - output.reconstruction) ** 2).mean()
                feature_residual = ((output.real_features - output.reconstruction_features) ** 2).mean()
                encoder_loss = image_residual if encoder_mode == "izi" else image_residual + kappa * feature_residual
        scaler.scale(encoder_loss).backward()
        scaler.step(encoder_optimizer)
        scaler.update()
        image_residual_value = float(image_residual.detach().cpu())
        feature_residual_value = float(feature_residual.detach().cpu())
        encoder_loss_value = float(encoder_loss.detach().cpu())
        if iteration % log_interval == 0 or iteration == 1:
            elapsed = max(1e-6, time.perf_counter() - encoder_start)
            global_iteration = wgan_iterations + iteration
            logger.info(
                "fastAnoGAN encoder iteration %s/%s mode=%s loss=%.5f image=%.5f feature=%.5f %.0f img/s",
                iteration,
                encoder_iterations,
                encoder_mode,
                encoder_loss_value,
                image_residual_value,
                feature_residual_value,
                iteration * batch_size / elapsed,
            )
            db.add(
                models.TrainingRunMetric(
                    training_run_id=run.id,
                    epoch=global_iteration,
                    train_loss=encoder_loss_value,
                    val_loss=feature_residual_value,
                )
            )
            run.epochs_completed = global_iteration
            run.train_loss = encoder_loss_value
            run.val_loss = feature_residual_value
            db.commit()

    torch.save(
        {
            "generator_state_dict": generator.state_dict(),
            "critic_state_dict": critic.state_dict(),
            "encoder_state_dict": encoder.state_dict(),
            "method_config": configuration.method_config,
            "method_graph": configuration.method_graph,
            "training_config": training_parameters,
            "feature_layer": configuration.method_graph.get("feature_layer", "critic_blocks"),
            "input_shape": [
                int(configuration.method_config["input_channels"]),
                int(configuration.method_config["input_height"]),
                int(configuration.method_config["input_width"]),
            ],
        },
        artifact_path,
    )
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
            artifact_dir = _run_artifact_dir(run_id)

            if configuration.builder_kind == "form":
                logger.info("Training run %s resolving training image paths", run_id)
                image_paths = enumerate_or_fail(db, pipeline)
                logger.info("Training run %s resolved %s training image paths", run_id, len(image_paths))
                # Fit-style methods (mean image) are pure numpy → always CPU.
                run.device = "CPU"
                db.commit()
                artifact_path = artifact_dir / "artifact.npy"
                count = fit_mean_image(
                    image_paths, graph, configuration.method_config, artifact_path, abort_event
                )
                run.artifact_kind = "mean_image"
            elif configuration.builder_kind == "spatiotemporal_autoencoder":
                logger.info("Training run %s resolving sequence clips", run_id)
                clips, skipped = enumerate_clips_or_fail(pipeline, configuration.method_config)
                logger.info("Training run %s resolved %s clips (%s skipped)", run_id, len(clips), skipped)
                artifact_path = artifact_dir / "artifact.pt"
                count = train_spatiotemporal_gradient(
                    db, run, configuration, clips, graph, run.training_parameters, artifact_path, abort_event
                )
                run.artifact_kind = "weights"
            elif configuration.builder_kind == "fast_anogan":
                logger.info("Training run %s resolving training image paths", run_id)
                image_paths = enumerate_or_fail(db, pipeline)
                logger.info("Training run %s resolved %s training image paths", run_id, len(image_paths))
                artifact_path = artifact_dir / "artifact.pt"
                count = train_fast_anogan(
                    db, run, configuration, image_paths, graph, run.training_parameters, artifact_path, abort_event
                )
                run.artifact_kind = "gan_bundle"
            else:
                logger.info("Training run %s resolving training image paths", run_id)
                image_paths = enumerate_or_fail(db, pipeline)
                logger.info("Training run %s resolved %s training image paths", run_id, len(image_paths))
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


def enumerate_clips_or_fail(pipeline: models.TrainingPipeline, method_config: dict):
    from app.training.data import enumerate_training_pipeline_clip_samples

    summary = enumerate_training_pipeline_clip_samples(pipeline, method_config)
    if not summary.clips:
        raise ValueError("No valid sequence clips found for this training pipeline's training sets.")
    return list(summary.clips), summary.skipped_missing
