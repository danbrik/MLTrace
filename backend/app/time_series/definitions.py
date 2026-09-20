"""Versioned model contracts and field-level provenance. No torch import at API startup."""
from copy import deepcopy
import math

USAD_SOURCE = "Audibert et al. (2020), equations 7–9, appendix A.3/A.4; SWaT profile"
TCN_SOURCE = "MarkusThill/bioma-tcn-ae, src/tcnae.py (BIOMA 2020 baseline)"
VAE_SOURCE = "Park et al. (2017), arXiv:1711.00614v1, sections III.C and IV.C"
SOURCES = {
    "usad": {"reference": USAD_SOURCE, "url": "https://doi.org/10.1145/3394486.3403392"},
    "tcn_ae": {"reference": TCN_SOURCE, "url": "https://github.com/MarkusThill/bioma-tcn-ae/blob/main/src/tcnae.py"},
    "lstm_vae": {"reference": VAE_SOURCE, "url": "https://arxiv.org/abs/1711.00614"},
}


def field(label, default, origin="MLTrace-Standard", reference="MLTrace time-series v1", minimum=None, maximum=None, note=""):
    return dict(label=label, default=default, origin=origin, reference=reference, minimum=minimum, maximum=maximum, adaptation=note)


def paper(label, default, ref, minimum=None, maximum=None):
    return field(label, default, "Paper", ref, minimum, maximum)


def author(label, default, minimum=None, maximum=None):
    return field(label, default, "Autorenimplementierung", TCN_SOURCE, minimum, maximum)


COMMON_TRAINING = {
    "seed": field("Seed", 42, minimum=0, maximum=4294967295),
    "shuffle": field("Trainingsfenster mischen", True),
}
COMMON_INPUT = {
    "nominal_history_minutes": field("Nominale Standardhistorie (min)", 180),
    "window_length": field("Fensterlänge bei 5 min Sampling", 36, note="Für alle Modelle gleich; Pipeline speichert L, nicht eine Dauer."),
    "step": field("Schrittweite (Samples)", 1),
    "gap_factor": field("Zeitlückengrenze × Train-Median", 1.5),
    "scaling": field("Skalierung", "minmax", note="Nur Train fitten; kein Clipping; konstanter Sensor: Nenner 1."),
}
DEFINITIONS = {
    "usad": {
        "label": "USAD", "version": "1", "source": SOURCES["usad"],
        "adaptations": ["Gemeinsames Fenster statt datensatzabhängiger Paper-Fenster; kein Downsampling.", "Feste Rekonstruktionsmetrik für Checkpoint-Auswahl; kein Test-Fitting."],
        "diagram": ["L × D → Flatten", "Encoder (ReLU)", "Latent z", "Decoder 1 / Decoder 2 (Sigmoid)", "AE1 und AE2(AE1)"],
        "architecture": {
            "latent_dim": paper("Latente Dimension", 40, USAD_SOURCE, 1, 4096),
            "hidden_ratio_1": paper("Encoderbreite 1 / Eingabebreite", 0.5, USAD_SOURCE, 0.01, 4),
            "hidden_ratio_2": paper("Encoderbreite 2 / Eingabebreite", 0.25, USAD_SOURCE, 0.01, 4),
        },
        "training": {
            "epochs": paper("Epochen", 70, USAD_SOURCE, 1, 100000),
            "lr": paper("Adam-Lernrate", 0.001, USAD_SOURCE + "; Adam default learning rate", 1e-8, 1),
            "batch_size": field("Batchgröße", 128, minimum=1, maximum=65536),
            "early_stopping": field("Early Stopping", False),
            "patience": field("Geduld (Epochen)", 10, minimum=1, maximum=100000),
            "alpha": field("Scoregewicht α (β = 1 − α)", 0.5, minimum=0, maximum=1),
        },
    },
    "tcn_ae": {
        "label": "TCN-AE · MLTrace-Anpassung", "version": "1", "source": SOURCES["tcn_ae"],
        "adaptations": ["Poolingfaktor 6 statt 42: bei L=36 ergibt sich ein Bottleneck 6 × 8.", "Partielle letzte Poolinggruppe; Upsampling-Ausgabe auf L zuschneiden.", "Direkter quadratischer Rekonstruktionsscore statt Test-Fitting und Glättung der Autorenimplementierung."],
        "diagram": ["L × D", "TCN-Encoder + Skip", "1×1-Projektion", "Average Pooling → z", "Upsampling + TCN-Decoder", "Lineare Rekonstruktion"],
        "architecture": {
            "filters": author("TCN-Filter", 20, 1, 1024),
            "kernel_size": author("Kernelgröße", 20, 1, 256),
            "dilations": author("Dilatationen", [1, 2, 4, 8, 16]),
            "stacks": author("TCN-Stacks", 1, 1, 8),
            "latent_channels": author("Bottleneck-Kanäle", 8, 1, 1024),
            "dropout": author("Dropout", 0.0, 0, 0.9),
            "pooling_factor": field("Poolingfaktor", 6, minimum=1, maximum=4096, note="Architekturanpassung: Autorenimplementierung verwendet 42."),
        },
        "training": {
            "epochs": author("Epochen", 40, 1, 100000), "batch_size": author("Batchgröße", 32, 1, 65536),
            "lr": author("Adam-Lernrate", 0.001, 1e-8, 1), "amsgrad": author("AMSGrad", True),
            "early_stopping": author("Early Stopping", False), "patience": author("Geduld (Epochen)", 2, 1, 100000),
        },
    },
    "lstm_vae": {
        "label": "LSTM-VAE · adaptierte Fenster-Variante", "version": "1", "source": SOURCES["lstm_vae"],
        "adaptations": ["Keine exakte Reproduktion von Park et al.: Sliding Windows mit Zustandsreset je Fenster.", "Standardnormalprior N(0,I) ersetzt den Roboterfortschrittsprior.", "Deterministische Auswertung und μ_z(X_t) nach Verarbeitung des vollständigen Fensters; keine Zufallsstichprobe.", "Kein Trainingsrauschen im Standard; Early Stopping anhand deterministischer Validation-ELBO."],
        "diagram": ["L × D (Zustandsreset)", "LSTM-Encoder", "μ_z / log σ²_z je Schritt", "Reparametrisierung (Training)", "LSTM-Decoder", "μ_x / σ²_x"],
        "architecture": {
            "latent_dim": paper("Latente Dimension", 3, VAE_SOURCE, 1, 4096),
            "hidden_size": field("LSTM Hidden Size", 64, minimum=1, maximum=4096),
            "layers": field("LSTM-Schichten", 1, minimum=1, maximum=8),
        },
        "training": {
            "epochs": field("Maximale Epochen", 100, minimum=1, maximum=100000),
            "batch_size": field("Batchgröße", 128, minimum=1, maximum=65536),
            "lr": paper("Adam-Lernrate", 0.001, VAE_SOURCE, 1e-8, 1),
            "early_stopping": field("Early Stopping auf Validation", True, note="Validation-Auswahl ist eine MLTrace-Anpassung."),
            "patience": paper("Geduld (Epochen)", 4, VAE_SOURCE, 1, 100000),
            "kl_weight": field("KL-Gewicht", 1.0, minimum=0, maximum=100),
            "noise_std": field("Trainingsrauschen (Standardabweichung)", 0.0, minimum=0, maximum=10),
        },
    },
}
for _definition in DEFINITIONS.values():
    _definition["training"].update(deepcopy(COMMON_TRAINING))
    _definition["input"] = deepcopy(COMMON_INPUT)


def definition(kind):
    if kind not in DEFINITIONS:
        raise ValueError("Unbekanntes Zeitreihenmodell.")
    return deepcopy({"kind": kind, **DEFINITIONS[kind]})


def defaults(kind, section):
    return {key: deepcopy(item["default"]) for key, item in definition(kind)[section].items()}


def validate_values(kind, section, values):
    fields = definition(kind)[section]
    if set(values) - set(fields):
        raise ValueError(f"Unbekannte {section}-Parameter: {', '.join(set(values) - set(fields))}")
    merged = defaults(kind, section) | values
    for key, spec in fields.items():
        value, base = merged[key], spec["default"]
        valid = False
        if isinstance(base, bool):
            valid = isinstance(value, bool)
        elif isinstance(base, list):
            valid = isinstance(value, list) and 0 < len(value) <= 12 and all(type(i) is int and 1 <= i <= 1024 for i in value)
        elif isinstance(base, int):
            valid = type(value) is int
        elif isinstance(base, float):
            valid = type(value) in (float, int) and math.isfinite(value)
        if not valid:
            raise ValueError(f"Ungültiger Wert für {spec['label']}.")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            if spec["minimum"] is not None and value < spec["minimum"] or spec["maximum"] is not None and value > spec["maximum"]:
                raise ValueError(f"{spec['label']}: Wert außerhalb des zulässigen Bereichs.")
    return merged


def provenance(kind, architecture, training=None):
    result = definition(kind)
    for section, values in (("architecture", architecture), ("training", training)):
        if values is not None:
            for key, spec in result[section].items():
                spec["value"] = values[key]
                spec["overridden"] = values[key] != spec["default"]
    return result

def representation_metadata(kind, length, config):
    if kind == "tcn_ae":
        shape = [math.ceil(length / config["pooling_factor"]), config["latent_channels"]]
        return dict(shape=shape, dimension=math.prod(shape), readout="pooled_encoder", flatten_order="time,channel", pooling_factor=config["pooling_factor"])
    return dict(shape=[config["latent_dim"]], dimension=config["latent_dim"],
                readout="full_window_final_posterior_mean" if kind == "lstm_vae" else "encoder_output",
                encoder_structure="per_step_posterior" if kind == "lstm_vae" else "flattened_window_mlp",
                prior="N(0,I)" if kind == "lstm_vae" else None)
