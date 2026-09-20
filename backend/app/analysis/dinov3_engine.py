"""Label-independent feature extraction and exploratory geometry.

Optional ML packages are imported only when executing an analysis.
"""
from __future__ import annotations

from bisect import bisect_left
import hashlib
import importlib.metadata
import json
import os
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from app.analysis.dinov3_schemas import RepresentationConfig, SelectionPreview

MODEL_ID = "timm/vit_small_patch16_dinov3.lvd1689m"
MODEL_REVISION = "3bf4720a82ec2066db88137180ff1f83a675cef0"
MODEL_ARCHITECTURE = "vit_small_patch16_dinov3"
LABELS = ("normal", "anomaly", "buffer")


def sample_records(records, config: RepresentationConfig):
    ordered = sorted({item.file_path: item for item in records}.values(),
                     key=lambda item: (item.timestamp_parsed, item.file_path))
    timestamps = [item.timestamp_parsed for item in ordered]
    samples, counts = [], []
    for interval in sorted(config.intervals, key=lambda item: (item.start, item.end, item.id)):
        candidates = ordered[bisect_left(timestamps, interval.start):bisect_left(timestamps, interval.end)]
        # Stable per-range random streams; neither labels nor UI order enter the seed.
        digest = hashlib.sha256(f"{config.seed}:{interval.id}".encode()).digest()
        rng = np.random.default_rng(int.from_bytes(digest[:16], "big"))
        rate = interval.sampling_rate
        for start in range(0, len(candidates) - rate + 1, rate):
            offset = int(rng.integers(rate)) if interval.random else rate - 1
            item = candidates[start + offset]
            samples.append({
                "file_path": item.file_path, "timestamp": item.timestamp_parsed.isoformat(),
                "interval_id": interval.id, "interval_name": interval.name,
                "label": interval.label, "event_id": interval.event_id,
            })
        counts.append({"id": interval.id, "name": interval.name, "label": interval.label,
                       "event_id": interval.event_id, "available": len(candidates),
                       "selected": len(candidates) // rate, "remainder": len(candidates) % rate})
    labels = Counter(row["label"] for row in samples)
    errors = [f"Bereich '{row['name']}' ergibt keine vollständige Stichprobe." for row in counts if not row["selected"]]
    if len(samples) < 4:
        errors.append("Mindestens vier ausgewählte Bilder sind erforderlich.")
    if not labels["normal"] or not labels["anomaly"]:
        errors.append("Normal- und Anomaliebeispiele sind erforderlich.")
    if config.cluster_count > len(samples):
        errors.append("Die Clusterzahl darf die Bildzahl nicht überschreiten.")
    preview = SelectionPreview(intervals=counts, label_counts={label: labels[label] for label in LABELS},
                               total=len(samples), errors=errors)
    return samples, preview


def unit_rgb(image: np.ndarray) -> np.ndarray:
    """Preserve absolute intensity using the dtype range, never image extrema."""
    array = np.asarray(image)
    if array.ndim == 2:
        array = array[..., None]
    if array.ndim != 3 or array.shape[2] not in (1, 3) or min(array.shape[:2]) < 1:
        raise ValueError("Die Pipeline muss ein Bild mit einem oder drei Kanälen erzeugen.")
    if np.issubdtype(array.dtype, np.integer):
        info = np.iinfo(array.dtype)
        result = ((array.astype(np.float64) - float(info.min)) / (float(info.max) - float(info.min))).astype(np.float32)
    elif np.issubdtype(array.dtype, np.floating):
        if not np.isfinite(array).all() or np.any(array < 0) or np.any(array > 1):
            raise ValueError("Float-Bilder müssen endlich sein und im Bereich [0, 1] liegen.")
        result = array.astype(np.float32)
    else:
        raise ValueError(f"Nicht unterstützter Bild-Datentyp: {array.dtype}.")
    return np.repeat(result, 3, axis=2) if result.shape[2] == 1 else result


def model_rgb(image: np.ndarray) -> np.ndarray:
    return cv2.resize(unit_rgb(image), (224, 224), interpolation=cv2.INTER_CUBIC).clip(0, 1)


class DinoEncoder:
    def __init__(self, device: str):
        try:
            import torch
            import timm
            from huggingface_hub import hf_hub_download
            from huggingface_hub.errors import LocalEntryNotFoundError
            from safetensors.torch import load_file
        except ImportError as exc:
            raise ValueError("DINOv3-Abhängigkeiten fehlen. Installiere MLTrace mit dem Extra 'ml'.") from exc
        self.torch = torch
        self.device = device
        from app.projects import ROOT_DIR
        cache_dir = os.environ.get("MLTRACE_MODEL_CACHE_DIR", str(ROOT_DIR / "model_cache" / "huggingface"))
        def download(filename):
            options = dict(repo_id=MODEL_ID, filename=filename, revision=MODEL_REVISION,
                           cache_dir=cache_dir, token=False)
            try:
                return hf_hub_download(**options, local_files_only=True)
            except LocalEntryNotFoundError:
                return hf_hub_download(**options)
        try:
            model_config = json.loads(Path(download("config.json")).read_text())
            weights_path = download("model.safetensors")
            download("LICENSE.md")
        except (OSError, ValueError) as exc:
            raise ValueError("Der öffentliche DINOv3-Download ist fehlgeschlagen. Netzwerk und lokalen Cache prüfen; ein Konto oder Token ist nicht erforderlich.") from exc
        if model_config.get("architecture") != MODEL_ARCHITECTURE:
            raise ValueError("Die gespeicherte DINOv3-Modellkonfiguration ist ungültig.")
        self.model = timm.create_model(MODEL_ARCHITECTURE, pretrained=False, img_size=224,
                                       num_classes=0, global_pool="token")
        self.model.load_state_dict(load_file(weights_path, device="cpu"), strict=True)
        # Match the original checkpoint's bfloat16 RoPE periods (see timm model card).
        self.model.rope.periods = self.model.rope.periods.to(torch.bfloat16).to(torch.float32)
        self.model.requires_grad_(False)
        self.model.eval()
        self.model.to(device)
        processor = model_config["pretrained_cfg"]
        self.mean = np.array(processor["mean"], dtype=np.float32)
        self.std = np.array(processor["std"], dtype=np.float32)
        self.snapshot = {
            "model_id": MODEL_ID, "revision": MODEL_REVISION, "processor": processor,
            "provider": "timm", "download_authentication": "anonymous",
            "rope_period_precision": "bfloat16", "architecture": MODEL_ARCHITECTURE,
            "input_size": [224, 224], "resize": "bicubic", "crop": False,
            "intensity": "dtype_range", "feature": "forward_features[:,0,:]",
            "feature_normalization": "none", "device": device,
            "versions": {"torch": torch.__version__, "timm": importlib.metadata.version("timm"),
                         "opencv": cv2.__version__},
        }

    def encode(self, images: list[np.ndarray]) -> np.ndarray:
        pixels = np.stack([(model_rgb(image) - self.mean) / self.std for image in images])
        tensor = self.torch.from_numpy(pixels.transpose(0, 3, 1, 2).copy()).to(self.device)
        with self.torch.inference_mode():
            output = self.model.forward_features(tensor)[:, 0, :]
        return output.detach().float().cpu().numpy()


def analyze_features(features: np.ndarray, config: RepresentationConfig, report):
    try:
        from sklearn.cluster import KMeans
        from sklearn.decomposition import PCA
        from umap import UMAP
    except ImportError as exc:
        raise ValueError("Analyse-Abhängigkeiten fehlen. Installiere MLTrace mit dem Extra 'ml'.") from exc
    values = np.asarray(features, dtype=np.float32)
    if values.ndim != 2 or min(values.shape) < 2 or len(values) < 4 or not np.isfinite(values).all():
        raise ValueError("Mindestens vier endliche Featurevektoren mit mindestens zwei Dimensionen sind erforderlich.")
    unique_count = len(np.unique(values, axis=0))
    if unique_count < 2:
        raise ValueError("Alle Featurevektoren sind konstant; eine Strukturanalyse ist nicht möglich.")
    if config.cluster_count > unique_count:
        raise ValueError("Die Clusterzahl übersteigt die Zahl unterschiedlicher Featurevektoren.")
    report("pca", 0, None)
    visual_pca = PCA(n_components=2, svd_solver="full", whiten=False)
    pca_xy = visual_pca.fit_transform(values)
    reduced_pca = PCA(n_components=config.pca_variance, svd_solver="full", whiten=False)
    reduced = reduced_pca.fit_transform(values)
    if len(np.unique(reduced, axis=0)) < config.cluster_count:
        raise ValueError("Nach PCA sind zu wenige unterschiedliche Punkte vorhanden. Varianzanteil erhöhen oder k reduzieren.")
    report("umap", 0, None)
    umap_xy = UMAP(n_components=2, n_neighbors=min(15, len(values) - 1), min_dist=0.1,
                   metric="euclidean", random_state=config.seed, n_jobs=1, init="random").fit_transform(values)
    report("kmeans", 0, None)
    clusters = KMeans(n_clusters=config.cluster_count, n_init=10, random_state=config.seed).fit_predict(reduced)
    if len(np.unique(clusters)) != config.cluster_count:
        raise ValueError("KMeans konnte die angeforderte Clusterzahl nicht bilden.")
    return pca_xy, umap_xy, clusters, {
        "sample_count": len(values), "feature_dimension": values.shape[1],
        "pca_2d_variance": visual_pca.explained_variance_ratio_.tolist(),
        "pca_components": int(reduced_pca.n_components_),
        "pca_retained_variance": float(reduced_pca.explained_variance_ratio_.sum()),
        "cluster_count": config.cluster_count,
    }


def evaluate_clusters(labels, clusters, metrics):
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    counts = Counter(labels)
    table = [{"cluster": index, **{label: sum(y == label and int(c) == index for y, c in zip(labels, clusters))
                                   for label in LABELS}}
             for index in range(metrics["cluster_count"])]
    return {**metrics, "ari": float(adjusted_rand_score(labels, clusters)),
            "nmi": float(normalized_mutual_info_score(labels, clusters)),
            "label_counts": {label: counts[label] for label in LABELS}, "contingency": table}
