import torch
import numpy as np
import os
import yaml
from ultralytics import YOLO  # type: ignore
from ultralytics.engine.results import Boxes, Results
from pathlib import Path


class YOLOImpl():
    """
    二次封装YOLO模型的类，提供训练、预测和跟踪功能。
    """

    def __init__(self, weights_path: str):
        torch.backends.cudnn.benchmark = True
        self.model = YOLO(Path(weights_path).absolute())
    

    def freeze(self, layer: int=10):
        freeze_layers = [f"model.{i}" for i in range(layer)]
        for name, param in self.model.named_parameters():
            if any(layer in name for layer in freeze_layers):
                param.requires_grad = False


    def train(
        self,
        data,
        epochs,
        batch=8,
        optimizer="AdamW",
        lr0=3e-4,
        pretrained=None,
        default_pretrained="./models/yolo11n.pt"
    ):
        if pretrained == "self":
            self.model.save(Path("./temp/detect/train_yolo/weights/best.pt").absolute())
            pretrained = "./temp/detect/train_yolo/weights/best.pt"
        elif pretrained is None:
            pretrained = default_pretrained

        self.model.train(
            augment=True,
            batch=batch,
            cache=False,
            compile=False,
            cutmix=0.2,
            data=data,
            device="0",
            dropout=0.1,
            epochs=epochs,
            exist_ok=True,
            freeze=10,
            hsv_s=0.5,
            hsv_v=0.3,
            lr0=lr0,
            mixup=0.2,
            name="train_yolo",
            optimizer=optimizer,
            perspective=0.0002,
            pretrained=None if pretrained is None else Path(pretrained).absolute(),
            project=Path("./temp/detect").absolute(),
            scale=0.7,
            workers=0,

            box=7.5,
            cls=3.0,
            dfl=1.5
        )
        self.model = YOLO(Path("./temp/detect/train_yolo/weights/best.pt").absolute())
    
    
    def predict(self, img):
        results = self.model(
            img,
            imgsz=640,
            conf=0.20,
            iou=0.6,
            half=True,
            device="0"
        )
        return results
    

    def track(self, img):
        config = {
            "tracker_type": "botsort",
            "track_high_thresh": 0.4,
            "track_low_thresh": 0.1,
            "new_track_thresh": 0.25,
            "track_buffer": 10,
            "match_thresh": 0.6,
            "fuse_score": True,
            "gmc_method": "sparseOptFlow",
            "proximity_thresh": 0.5,
            "appearance_thresh": 0.8,
            "with_reid": False,
            "model": "auto"
        }
        os.makedirs("./temp/track", exist_ok=True)
        yaml.dump(config, open("./temp/track/bytetrack.yaml", "w", encoding="utf-8"), indent=4)
        results = self.model.track(
            img,
            rect=True,
            half=True,
            persist=False,
            tracker=Path("./temp/track/bytetrack.yaml").absolute(),
            device="0",
            augment=True,
        )

        for e in results:  # 去除未分配追踪ID的目标框
            if e.boxes is None:
                continue
            data = e.boxes.data
            data = data if isinstance(data, torch.Tensor) else torch.tensor(data)
            if data is None:
                empty = torch.zeros((0, 7), dtype=torch.float32)
                e.boxes = Boxes(empty, e.boxes.orig_shape)
                continue
            if data.numel() == 0:
                empty = torch.zeros((0, 7), dtype=data.dtype, device=data.device)
                e.boxes = Boxes(empty, e.boxes.orig_shape)
                continue
            if data.shape[1] == 6:
                empty = torch.zeros((0, 7), dtype=data.dtype, device=data.device)
                e.boxes = Boxes(empty, e.boxes.orig_shape)
                continue
            if data.shape[1] == 7:
                ids = data[:, 4]
                mask = torch.isfinite(ids) & (ids > 0)
                data = data[mask]
                if data.numel() == 0:
                    empty = torch.zeros((0, 7), dtype=ids.dtype, device=ids.device)
                    e.boxes = Boxes(empty, e.boxes.orig_shape)
                else:
                    e.boxes = Boxes(data, e.boxes.orig_shape)

        return results