SSIM_LOSS_OPTIONS = ["mse", "l1", "smooth_l1", "ssim", "mae_ssim", "mse_ssim"]
SSIM_ERROR_METRIC_OPTIONS = ["mse", "mae", "ssim_distance"]

SSIM_TRAINING_PROPERTIES = {
    "ssim_window_size": {
        "type": "integer",
        "label": "SSIM window size",
        "minimum": 3,
        "default": 11,
        "description": "Odd local window size for SSIM. 11 is the standard setting used in many SSIM autoencoder experiments.",
    },
    "ssim_alpha": {"type": "number", "label": "SSIM alpha", "minimum": 0, "default": 1.0},
    "ssim_beta": {"type": "number", "label": "SSIM beta", "minimum": 0, "default": 1.0},
    "ssim_gamma": {"type": "number", "label": "SSIM gamma", "minimum": 0, "default": 1.0},
    "ssim_k1": {
        "type": "number",
        "label": "SSIM K1",
        "minimum": 0,
        "default": 0.01,
        "description": "Standard SSIM K constant. MLTrace computes C1=(K1*data_range)^2.",
    },
    "ssim_k2": {
        "type": "number",
        "label": "SSIM K2",
        "minimum": 0,
        "default": 0.03,
        "description": "Standard SSIM K constant. MLTrace computes C2=(K2*data_range)^2.",
    },
    "ssim_data_range": {
        "type": "number",
        "label": "SSIM data range",
        "minimum": 0,
        "default": 1.0,
        "description": "Expected value range of model inputs/reconstructions. Use 1.0 for normalized float images.",
    },
    "ssim_weight": {
        "type": "number",
        "label": "SSIM weight",
        "minimum": 0,
        "maximum": 1,
        "default": 0.5,
        "description": "Only used by mae_ssim and mse_ssim. 0 uses only pixel loss; 1 uses only SSIM loss.",
    },
}

SSIM_INFERENCE_PROPERTIES = {
    key: value for key, value in SSIM_TRAINING_PROPERTIES.items() if key != "ssim_weight"
}
