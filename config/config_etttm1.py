ETTM1_CONFIG = {
    "data_path": "ETTm1",
    "seq_len": 96,
    "batch_size": 16,
    "runs": [
        {"pred_len": 96,  "learning_rate": 1e-4, "channel": 64, "e_layer": 2, "d_layer": 2, "dropout_n": 0.5, "epochs": 100, "seed": 2024},
        {"pred_len": 192, "learning_rate": 1e-4, "channel": 64, "e_layer": 2, "d_layer": 2, "dropout_n": 0.5, "epochs": 100, "seed": 2024},
        {"pred_len": 336, "learning_rate": 1e-4, "channel": 64, "e_layer": 2, "d_layer": 2, "dropout_n": 0.5, "epochs": 100, "seed": 2024},
        {"pred_len": 720, "learning_rate": 1e-4, "channel": 64, "e_layer": 2, "d_layer": 2, "dropout_n": 0.7, "epochs": 100, "seed": 2024},
    ],
}
