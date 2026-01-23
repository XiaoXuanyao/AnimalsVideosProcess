import torch
import numpy as np
import os
import yaml
from ultralytics import YOLO  # type: ignore
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
        optimizer="Adam",
        lr=3e-4,
        pretrained=None,
        default_pretrained="./models/yolo11n.pt"
    ):
        if pretrained == "self":
            self.model.save(Path("./temp/detect/train_yolo/weights/best.pt").absolute())
            pretrained = "./temp/detect/train_yolo/weights/best.pt"
        elif pretrained is None:
            pretrained = "./models/yolo11n.pt"

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
            lr0=lr,
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
            conf=0.18,
            iou=0.5,
            half=True,
            device="0"
        )
        return results
    

    def track(self, img, detects):
        predictor = self.model.predictor
        if predictor is not None:
            trackers = getattr(predictor, "trackers", None)
            if isinstance(trackers, list) and len(trackers) > 0:
                tracker = trackers[0]
                tracker.reset()

                class _DetObj:
                    def __init__(self, arr=None, xyxy=None, xywh=None, conf=None, cls=None):
                        def _xyxy_to_xywh(xyxy):
                            a = np.asarray(xyxy, dtype=np.float32)
                            if a.size == 0:
                                return np.empty((0, 4), dtype=np.float32)
                            if a.ndim == 1:
                                a = a.reshape(1, 4)
                            x1, y1, x2, y2 = a.T
                            w = x2 - x1
                            h = y2 - y1
                            cx = x1 + w / 2.0
                            cy = y1 + h / 2.0
                            return np.stack([cx, cy, w, h], axis=1).astype(np.float32)

                        if arr is not None:
                            a = np.asarray(arr, dtype=np.float32)
                            if a.ndim == 1 and a.size > 0:
                                a = a.reshape(1, -1)
                            if a.size == 0:
                                self.xyxy = np.empty((0, 4), dtype=np.float32)
                                self.conf = np.empty((0,), dtype=np.float32)
                                self.cls = np.empty((0,), dtype=np.int32)
                                self.xywh = np.empty((0, 4), dtype=np.float32)
                                self.xywhr = self.xywh
                            else:
                                self.xyxy = a[:, :4].astype(np.float32)
                                self.conf = a[:, 4].astype(np.float32) if a.shape[1] > 4 else np.zeros((a.shape[0],), dtype=np.float32)
                                self.cls = a[:, 5].astype(np.int32) if a.shape[1] > 5 else np.zeros((a.shape[0],), dtype=np.int32)
                                self.xywh = _xyxy_to_xywh(self.xyxy)
                                self.xywhr = self.xywh
                        else:
                            self.xyxy = np.asarray(xyxy, dtype=np.float32)
                            if self.xyxy.ndim == 1:
                                self.xyxy = self.xyxy.reshape(1, 4)
                            self.conf = np.asarray(conf, dtype=np.float32)
                            if self.conf.ndim == 0:
                                self.conf = self.conf.reshape(1,)
                            self.cls = np.asarray(cls, dtype=np.int32)
                            if self.cls.ndim == 0:
                                self.cls = self.cls.reshape(1,)
                            self.xywh = _xyxy_to_xywh(self.xyxy)
                            self.xywhr = self.xywh

                    def __len__(self):
                        return int(self.xyxy.shape[0])

                    def __getitem__(self, idx):
                        return _DetObj(xyxy=self.xyxy[idx], conf=self.conf[idx], cls=self.cls[idx])

                for e in detects:
                    det_obj = _DetObj(arr=e)
                    tracker.update(det_obj, None, None)

        config = {
            "tracker_type": "bytetrack",
            "track_high_thresh": 0.25,
            "track_low_thresh": 0.1,
            "new_track_thresh": 0.25,
            "track_buffer": 10,
            "match_thresh": 0.65,
            "fuse_score": True,
        }
        os.makedirs("./temp/track", exist_ok=True)
        yaml.dump(config, open("./temp/track/bytetrack.yaml", "w", encoding="utf-8"), indent=4)
        results = self.model.track(
            img,
            conf=0.18,
            iou=0.5,
            rect=True,
            half=True,
            persist=False,
            tracker=Path("./temp/track/bytetrack.yaml").absolute(),
            device="0",
            augment=True,
        )
        return results