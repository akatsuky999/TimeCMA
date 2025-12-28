import os
import time
import argparse
import random
from typing import Tuple

import torch
import h5py
from torch.utils.data import DataLoader

from data_provider.data_loader_save import (
    Dataset_ETT_hour,
    Dataset_ETT_minute,
    Dataset_Custom,
    Dataset_ST,
)
from storage.gen_pic_emb import GenPicEmb

# Default prompt for DeepSeek-OCR time series embedding
DEFAULT_PROMPT = (
    "<image>\nThis is a image containing multivariate time series information. "
    "The upper part of the image introduces the sample data."
    "The lower part is a heatmap visualization. "
    "The horizontal axis of the heatmap represents the temporal dimension."
    "The vertical axis of the heatmap represents the variable dimension."
    "Analyze temporal patterns and extract features from a joint temporal and spatio-variable perspective."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda", help="device id or 'cpu'")
    parser.add_argument("--data_path", type=str, default="ETTh1")
    parser.add_argument("--num_nodes", type=int, default=7)
    parser.add_argument("--input_len", type=int, default=96)
    parser.add_argument("--output_len", type=int, default=96)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--d_model", type=int, default=1280)  # DeepSeek-OCR native dim
    parser.add_argument("--model_name", type=str, default="deepseek-ai/DeepSeek-OCR")
    parser.add_argument("--divide", type=str, default="train", choices=["train", "test", "val"])
    parser.add_argument(
        "--num_workers",
        type=int,
        default=min(10, os.cpu_count() or 1),
        help="dataloader workers",
    )
    parser.add_argument(
        "--fixed_width",
        type=int,
        default=512,
        help="fixed width for heatmap image (time axis)",
    )
    parser.add_argument(
        "--cmap",
        type=str,
        default="viridis",
        help="matplotlib colormap for heatmap (e.g., viridis, plasma, inferno)",
    )
    parser.add_argument(
        "--add_text",
        action="store_true",
        help="add text description above heatmap image",
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="normalize each feature to [0,1] before plotting heatmap",
    )
    parser.add_argument(
        "--base_size",
        type=int,
        default=512,
        help="DeepSeek-OCR base_size config",
    )
    parser.add_argument(
        "--image_size_model",
        type=int,
        default=512,
        help="DeepSeek-OCR image_size config",
    )
    parser.add_argument(
        "--crop_mode",
        action="store_true",
        help="use cropping mode in DeepSeek-OCR (Gundam / dynamic resolution)",
    )
    parser.add_argument(
        "--save_root",
        type=str,
        default="./Embeddings_OCR",
        help="root directory to save OCR-based embeddings (default: ./Embeddings_OCR)",
    )
    # Legacy parameter for backward compatibility
    parser.add_argument(
        "--image_size",
        type=int,
        default=None,
        help="(deprecated) use --fixed_width instead",
    )
    # 图片保存选项
    parser.add_argument(
        "--save_images",
        action="store_true",
        help="randomly save some generated images for inspection",
    )
    parser.add_argument(
        "--save_image_ratio",
        type=float,
        default=0.001,
        help="ratio of images to save (default: 0.001 = 0.1%%)",
    )
    parser.add_argument(
        "--save_image_dir",
        type=str,
        default="./Debug_Images",
        help="directory to save debug images",
    )
    return parser.parse_args()


def get_dataset(data_path: str, flag: str, input_len: int, output_len: int):
    datasets = {
        "ETTh1": Dataset_ETT_hour,
        "ETTh2": Dataset_ETT_hour,
        "ETTm1": Dataset_ETT_minute,
        "ETTm2": Dataset_ETT_minute,
        # 5min STdata (ETT-like format)
        "STdata": Dataset_ST,
        # PEMS04: use Dataset_ST (5min, no time column) + file auto-detect (.csv/.npy/.npz)
        "PEMS04": Dataset_ST,
    }
    dataset_class = datasets.get(data_path, Dataset_Custom)
    return dataset_class(flag=flag, size=[input_len, 0, output_len], data_path=data_path)


def save_embeddings_ocr(args: argparse.Namespace) -> None:
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    train_set = get_dataset(args.data_path, "train", args.input_len, args.output_len)
    test_set = get_dataset(args.data_path, "test", args.input_len, args.output_len)
    val_set = get_dataset(args.data_path, "val", args.input_len, args.output_len)

    data_loader = {
        "train": DataLoader(
            train_set,
            batch_size=args.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=args.num_workers,
        ),
        "test": DataLoader(
            test_set,
            batch_size=args.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=args.num_workers,
        ),
        "val": DataLoader(
            val_set,
            batch_size=args.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=args.num_workers,
        ),
    }[args.divide]

    # Determine fixed_width: prefer --fixed_width, fallback to --image_size
    fixed_width = args.fixed_width
    if args.image_size is not None:
        fixed_width = args.image_size

    gen_pic_emb = GenPicEmb(
        data_path=args.data_path,
        model_name=args.model_name,
        device=str(device),
        input_len=args.input_len,
        d_model=args.d_model,
        fixed_width=fixed_width,
        cmap=args.cmap,
        add_text=args.add_text,
        normalize=args.normalize,
        divide=args.divide,
        base_size=args.base_size,
        image_size_model=args.image_size_model,
        crop_mode=args.crop_mode,
        prompt=DEFAULT_PROMPT,  # Explicitly pass prompt
    ).to(device)

    save_path = os.path.join(args.save_root, args.data_path, args.divide)
    os.makedirs(save_path, exist_ok=True)

    emb_time_path = "./Results/emb_logs_ocr/"
    os.makedirs(emb_time_path, exist_ok=True)

    # 设置图片保存目录
    image_save_path = None
    if args.save_images:
        image_save_path = os.path.join(args.save_image_dir, args.data_path, args.divide)
        os.makedirs(image_save_path, exist_ok=True)
        print(f"[Debug] 将随机保存 {args.save_image_ratio * 100:.2f}% 的图片到: {image_save_path}")

    saved_count = 0
    for i, (x, y, x_mark, y_mark) in enumerate(data_loader):
        # x: [B, S, N], x_mark: [B, S, D_time]
        x = x.to(device)
        x_mark = x_mark.to(device)

        # 决定是否保存这个 batch 的图片
        should_save_image = args.save_images and random.random() < args.save_image_ratio

        # Pass x_mark for real timestamp formatting in image text description
        embeddings, images = gen_pic_emb.generate_embeddings(
            x, in_data_mark=x_mark, return_images=should_save_image
        )
        # embeddings: [B, d_model, N]

        for b in range(embeddings.shape[0]):
            sample_idx = i * args.batch_size + b
            file_path = os.path.join(save_path, f"{sample_idx}.h5")
            with h5py.File(file_path, "w") as hf:
                hf.create_dataset("embeddings", data=embeddings[b].cpu().numpy())

            if should_save_image and images is not None and b < len(images):
                img_file = os.path.join(image_save_path, f"sample_{sample_idx}.png")
                images[b].save(img_file)
                saved_count += 1

    if args.save_images:
        print(f"[Debug] 共保存了 {saved_count} 张图片到 {image_save_path}")


if __name__ == "__main__":
    args = parse_args()
    t1 = time.time()
    save_embeddings_ocr(args)
    t2 = time.time()
    print(f"Total time spent (DeepSeek-OCR embeddings): {(t2 - t1) / 60:.4f} minutes")


