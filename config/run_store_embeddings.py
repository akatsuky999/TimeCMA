import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("PYTHONPATH", str(ROOT) + os.pathsep + os.environ.get("PYTHONPATH", ""))


def run_etttm1():
    data_paths = ["ETTm1"]
    divides = ["train", "val", "test"]
    num_nodes = 7
    input_len = 96
    output_len = 96

    for data_path in data_paths:
        for divide in divides:
            log_dir = ROOT / "Results" / "emb_logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            log_file = log_dir / f"{data_path}_{divide}.log"

            cmd = [
                "python", "storage/store_emb.py",
                "--divide", divide,
                "--data_path", data_path,
                "--num_nodes", str(num_nodes),
                "--input_len", str(input_len),
                "--output_len", str(output_len),
            ]
            print("Running:", " ".join(cmd))
            with open(log_file, "w", encoding="utf-8") as f:
                subprocess.run(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT, check=True)


def run_ili():
    data_paths = ["ILI"]
    divides = ["train", "val", "test"]
    num_nodes = 7
    input_len = 36
    output_len = 24

    for data_path in data_paths:
        for divide in divides:
            log_dir = ROOT / "Results" / "emb_logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            log_file = log_dir / f"{data_path}_{divide}.log"

            cmd = [
                "python", "storage/store_emb.py",
                "--divide", divide,
                "--data_path", data_path,
                "--num_nodes", str(num_nodes),
                "--input_len", str(input_len),
                "--output_len", str(output_len),
            ]
            print("Running:", " ".join(cmd))
            with open(log_file, "w", encoding="utf-8") as f:
                subprocess.run(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT, check=True)


if __name__ == "__main__":
    # run_ili()
    run_etttm1()
