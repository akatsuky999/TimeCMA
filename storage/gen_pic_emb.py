"""
Generate embeddings from multivariate time series via DeepSeek-OCR (vLLM version).

This module converts time series data to heatmap images and uses DeepSeek-OCR as an
image encoder to extract embeddings, similar to how GenPromptEmb uses GPT-2.

DeepSeek-OCR Pipeline:
    1. Image Preprocessing (DeepseekOCRProcessor)
    2. Vision Encoder (SAM + ViT)
    3. Feature Fusion + Projection → Vision Tokens
    4. Merge Vision Tokens with Text Tokens
    5. LLM Transformer Layers
    6. Pooling (last hidden state) → Embedding [1280]

Dependencies:
    - Requires DeepSeek-OCR folder at: ../../DeepSeek-OCR/DeepSeek-OCR-master/DeepSeek-OCR-vllm
    - If missing, ImportError is raised at module load time
"""
import os
import sys
from typing import Optional, List, Tuple, Union

import torch
import torch.nn as nn
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib

matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

# --- Setup vLLM environment ---
if torch.version.cuda == '11.8':
    os.environ["TRITON_PTXAS_PATH"] = "/usr/local/cuda-11.8/bin/ptxas"
os.environ['VLLM_USE_V1'] = '0'

# --- Add DeepSeek-OCR vLLM directory to path ---
_THIS_DIR = os.path.abspath(os.path.dirname(__file__))
DS_VLLM_ROOT = os.path.abspath(
    os.path.join(_THIS_DIR, "..", "..", "DeepSeek-OCR", "DeepSeek-OCR-master", "DeepSeek-OCR-vllm")
)
if not os.path.isdir(DS_VLLM_ROOT):
    raise ImportError(
        f"DeepSeek-OCR vLLM path not found: {DS_VLLM_ROOT}\n"
        "Please ensure the DeepSeek-OCR repository exists at the expected location."
    )
if DS_VLLM_ROOT not in sys.path:
    sys.path.insert(0, DS_VLLM_ROOT)

# --- vLLM and DeepSeek-OCR imports ---
from vllm import LLM, PoolingParams  # type: ignore
from vllm.model_executor.models.registry import ModelRegistry  # type: ignore
from deepseek_ocr import DeepseekOCRForCausalLM  # type: ignore
from process.image_process import DeepseekOCRProcessor  # type: ignore

# Register the custom model architecture
ModelRegistry.register_model("DeepseekOCRForCausalLM", DeepseekOCRForCausalLM)

from storage.T2I.TS2image import TimeSeriesToImage


class GenPicEmb(nn.Module):
    """
    Generate embeddings from multivariate time series via DeepSeek-OCR.

    Pipeline:
        Input [B, S, N] → Heatmap Image → DeepSeek-OCR encode → [B, 1280]
        → Broadcast to [B, 1280, N]

    Output shape [B, d_model, N] matches GenPromptEmb for compatibility.
    DeepSeek-OCR's native embedding dimension is 1280.
    """

    # DeepSeek-OCR native embedding dimension (from projector output)
    DSOCR_EMBED_DIM = 1280

    def __init__(
            self,
            data_path: str = "ETTh1",
            model_name: str = "deepseek-ai/DeepSeek-OCR",
            device: str = "cuda:0",
            d_model: int = 1280,  # Default to DeepSeek-OCR native dimension
            fixed_width: int = 512,
            cmap: str = "viridis",
            add_text: bool = True,  # Default True for embedding generation with text context
            normalize: bool = True,
            crop_mode: bool = False,
            max_model_len: int = 8192,
            gpu_memory_utilization: float = 0.8,
            prompt: Optional[str] = None,
            # Legacy/unused parameters kept for backward compatibility
            input_len: int = 96,
            divide: str = "train",
            base_size: int = 512,
            image_size_model: int = 512,
            image_size: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.data_path = data_path
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.d_model = d_model
        self.crop_mode = crop_mode

        # Handle legacy image_size parameter
        if image_size is not None:
            fixed_width = image_size

        # Time series → heatmap converter
        self.ts_to_image = TimeSeriesToImage(
            fixed_width=fixed_width, cmap=cmap, add_text=add_text,
            data_path=data_path, normalize=normalize
        )

        # Default prompt for DeepSeek-OCR
        self.prompt = prompt or (
            "<image>\nThis is a image containing multivariate time series information. "
            "The upper part of the image introduces the sample data."
            "The lower part is a heatmap visualization. "
            "The horizontal axis of the heatmap represents the temporal dimension."
            "The vertical axis of the heatmap represents the variable dimension."
            "Analyze temporal patterns and extract features from a joint temporal and spatio-variable perspective."
        )

        # Log prompt confirmation
        print("=" * 70)
        print("[GenPicEmb] Prompt configured successfully:")
        print("-" * 70)
        print(self.prompt)
        print("=" * 70)

        # Initialize vLLM with embed task
        # Parameters aligned with source project (run_dpsk_ocr_pdf.py)
        self.pooling_params = PoolingParams()
        self.llm = LLM(
            model=model_name,
            hf_overrides={"architectures": ["DeepseekOCRForCausalLM"]},
            trust_remote_code=True,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            block_size=256,  # From source project
            enforce_eager=False,  # From source project
            task="embed",  # For embedding extraction (not text generation)
        )
        self.processor = DeepseekOCRProcessor()

        # Projection layer: only needed if d_model != DSOCR_EMBED_DIM (1280)
        self.proj: Optional[nn.Linear] = None
        if d_model != self.DSOCR_EMBED_DIM:
            self.proj = nn.Linear(self.DSOCR_EMBED_DIM, d_model, bias=False)

    @torch.no_grad()
    def _encode_image(self, image: Image.Image) -> torch.Tensor:
        """
        Encode image through DeepSeek-OCR pipeline to get embedding.

        Full pipeline: Image → Preprocess → SAM+ViT → Projector →
                      Merge with Text → LLM Layers → Pool → [1280]
        """
        inputs = [{
            "prompt": self.prompt,
            "multi_modal_data": {
                "image": self.processor.tokenize_with_images(
                    images=[image], bos=True, eos=True, cropping=self.crop_mode
                )
            },
        }]
        outputs = self.llm.encode(inputs, pooling_params=self.pooling_params, use_tqdm=False)
        return outputs[0].outputs.data.to(self.device)

    @torch.no_grad()
    def generate_embeddings(
            self,
            in_data: torch.Tensor,
            in_data_mark: Optional[torch.Tensor] = None,
            return_images: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Optional[List[Image.Image]]]]:
        """
        Generate embeddings from multivariate time series.

        Args:
            in_data: [B, S, N] time series data
            in_data_mark: Optional [B, S, D_time] timestamps
            return_images: If True, also return the generated images for debugging

        Returns:
            If return_images=False: embeddings [B, d_model, N]
            If return_images=True: (embeddings [B, d_model, N], List[PIL.Image])

        Note:
            DeepSeek-OCR outputs [B, 1280] embeddings natively.
            If d_model == 1280 (default), no projection is applied.
        """
        if in_data.dim() != 3:
            raise ValueError(f"Expected in_data shape [B, S, N], got {in_data.shape}")

        B, S, N = in_data.shape

        # Encode each sample
        emb_list = []
        img_list = [] if return_images else None
        for b in range(B):
            data_mark_b = in_data_mark[b] if in_data_mark is not None else None
            img = self.ts_to_image.to_image(in_data[b], data_mark=data_mark_b)
            emb_list.append(self._encode_image(img))
            if return_images:
                img_list.append(img)

        # Stack embeddings: [B, 1280]
        emb_batch = torch.stack(emb_list, dim=0).to(self.device)

        # Ensure float32
        if emb_batch.dtype != torch.float32:
            emb_batch = emb_batch.float()

        # Project to d_model only if different from native 1280
        if self.proj is not None:
            self.proj = self.proj.to(dtype=emb_batch.dtype, device=self.device)
            emb_batch = self.proj(emb_batch)

        # Broadcast: [B, d_model] -> [B, d_model, N]
        embeddings = emb_batch.unsqueeze(-1).expand(B, self.d_model, N)

        if return_images:
            return embeddings, img_list
        return embeddings, None
