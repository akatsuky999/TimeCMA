import os
import subprocess
from pathlib import Path

from config_ili import ILI_CONFIG

ROOT = Path(__file__).resolve().parents[1]


def main():
    os.environ.setdefault("PYTHONPATH", str(ROOT) + os.pathsep + os.environ.get("PYTHONPATH", ""))
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    cfg = ILI_CONFIG
    data_path = cfg["data_path"]
    seq_len = cfg["seq_len"]
    batch_size = cfg["batch_size"]

    log_dir = ROOT / "Results" / data_path
    log_dir.mkdir(parents=True, exist_ok=True)

    for run in cfg["runs"]:
        pred_len = run["pred_len"]
        learning_rate = run["learning_rate"]
        channel = run["channel"]
        e_layer = run["e_layer"]
        d_layer = run["d_layer"]
        dropout_n = run["dropout_n"]
        epochs = run["epochs"]
        seed = run["seed"]
        num_workers = run.get("num_workers")
        d_llm = run.get("d_llm")

        log_file = log_dir / f"i{seq_len}_o{pred_len}_lr{learning_rate}_c{channel}_el{e_layer}_dl{d_layer}_dn{dropout_n}_bs{batch_size}.log"
        cmd = [
            "python", "train.py",
            "--data_path", data_path,
            "--batch_size", str(batch_size),
            "--num_nodes", "7",
            "--seq_len", str(seq_len),
            "--pred_len", str(pred_len),
            "--epochs", str(epochs),
            "--seed", str(seed),
            "--channel", str(channel),
            "--learning_rate", str(learning_rate),
            "--dropout_n", str(dropout_n),
            "--e_layer", str(e_layer),
            "--d_layer", str(d_layer),
        ]

        if num_workers is not None:
            cmd += ["--num_workers", str(num_workers)]
        if d_llm is not None:
            cmd += ["--d_llm", str(d_llm)]

        print("Running:", " ".join(cmd))
        with open(log_file, "w", encoding="utf-8") as f:
            subprocess.run(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT, check=True)


if __name__ == "__main__":
    main()
