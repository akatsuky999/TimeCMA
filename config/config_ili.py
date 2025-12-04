ILI_CONFIG = {
    "data_path": "ILI",
    "seq_len": 36,
    "batch_size": 16,
    "runs": [
        {"pred_len": 24, "learning_rate": 1e-4,  "channel": 16, "e_layer": 1, "d_layer": 1, "dropout_n": 0.1, "epochs": 100, "seed": 6666, "num_workers": 10, "d_llm": 768},
        {"pred_len": 36, "learning_rate": 1e-4,  "channel": 16, "e_layer": 1, "d_layer": 1, "dropout_n": 0.1, "epochs": 150, "seed": 6666},
        {"pred_len": 48, "learning_rate": 2.5e-3,"channel": 16, "e_layer": 1, "d_layer": 1, "dropout_n": 0.3, "epochs": 100, "seed": 8888},
        {"pred_len": 60, "learning_rate": 1e-4,  "channel": 32, "e_layer": 1, "d_layer": 1, "dropout_n": 0.1, "epochs": 100, "seed": 8888},
    ],
}
