import torch
import torch.nn.functional as F
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import typing
import tensorrt as trt
import torch_tensorrt
from transformers import pipeline, CLIPProcessor, CLIPModel
from PIL import Image
from pathlib import Path


LOCAL_MODEL_DIR = r"models\clip-vit-base-patch32"


class CLIPVisionWrapper(torch.nn.Module):
    def __init__(
            self,
            clip_model: CLIPModel
    ) -> None:
        super(CLIPVisionWrapper, self).__init__()
        self.clip_model = clip_model
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        img_embs = self.clip_model.get_image_features(pixel_values=x)  #type: ignore
        img_embs = img_embs / img_embs.norm(dim=-1, keepdim=True)
        return img_embs


class CLIP:
    def __init__(
            self,
            batch_size: int = 1,
            use_tensorrt: bool = True
    ) -> None:

        if not os.path.exists(LOCAL_MODEL_DIR):
            os.makedirs(LOCAL_MODEL_DIR, exist_ok=True)
            print("Downloading model and processor from Hugging Face and saving to", LOCAL_MODEL_DIR)
            model = CLIPModel.from_pretrained(
                "openai/clip-vit-base-patch32"
            )
            processor = CLIPProcessor.from_pretrained(
                "openai/clip-vit-base-patch32",
                use_fast=True
            )
            model.save_pretrained(LOCAL_MODEL_DIR)
            processor.save_pretrained(LOCAL_MODEL_DIR)
            print("OK.")
        else:
            print("Local model found:", LOCAL_MODEL_DIR)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = CLIPProcessor.from_pretrained(
            LOCAL_MODEL_DIR,
            local_files_only=True,
            use_fast=True
        )
            
        self.model = CLIPModel.from_pretrained(
            LOCAL_MODEL_DIR,
            local_files_only=True
        )
        self.model.to(self.device)  # type: ignore
        self.model.eval()
            
        if use_tensorrt and os.path.exists(Path(LOCAL_MODEL_DIR).parent / "clip_trt.pt2"):
            self.vision_trtmodel = torch_tensorrt.load(
                str(Path(LOCAL_MODEL_DIR).parent / "clip_trt.pt2")
            ).module().to(self.device)
            print("Using pre-built TensorRT optimized CLIP vision model.")
            return
            
        self.vision_trtmodel = CLIPVisionWrapper(self.model).to(self.device).eval()
        if use_tensorrt:
            print("Optimizing CLIP vision model with TensorRT...")
            self.vision_trtmodel = torch_tensorrt.compile(
                self.vision_trtmodel,
                inputs=[torch_tensorrt.Input(
                    min_shape=(1, 3, 224, 224),
                    opt_shape=(batch_size, 3, 224, 224),
                    max_shape=(batch_size, 3, 224, 224),
                    dtype=torch.float32
                )],
                enabled_precisions={torch.float16},
                workspace_size=1 << 30
            )

            torch_tensorrt.save(self.vision_trtmodel, str(Path(LOCAL_MODEL_DIR).parent / "clip_trt.pt2"))
            print("Using TensorRT optimized CLIP vision model.")
    

    def calculate_image(
            self,
            images: np.ndarray,
    ) -> np.ndarray:
        if len(images.shape) == 3:
            images = np.expand_dims(images, axis=0)
        img_inputs = self.processor(
            images=images,
            return_tensors="pt"     # type: ignore
        )
        img_inputs = {k: v.to(self.device) for k, v in img_inputs.items()}
        with torch.no_grad():
            img_embs = self.vision_trtmodel(img_inputs["pixel_values"])  # (1, D)
        return img_embs.cpu().numpy()
    

    def calculate_text(
            self,
            texts: list[str]
    ) -> np.ndarray:
        text_inputs = self.processor(
            text=texts,
            return_tensors="pt",    # type: ignore
            padding=True            # type: ignore
        )
        text_inputs = {k: v.to(self.device) for k, v in text_inputs.items()}
        with torch.no_grad():
            txt_emb = self.model.get_text_features(**text_inputs)  # (1, D)
            txt_emb = txt_emb / txt_emb.norm(dim=-1, keepdim=True)
        return txt_emb.cpu().numpy()