import torch
import numpy as np
from ultralytics import YOLO  # type: ignore
from pathlib import Path


class YOLO11n():

    def __init__(self, weights_path: str):
        torch.backends.cudnn.benchmark = True
        self.model = YOLO(Path(weights_path))
    

    def freeze(self):
        freeze_layers = [f"model.{i}" for i in range(8)]
        for name, param in self.model.named_parameters():
            if any(layer in name for layer in freeze_layers):
                param.requires_grad = False


    def train(self, data, epochs, batch=16):
        self.freeze()
        self.model.train(
            data=data,
            epochs=epochs,
            batch=batch,
            project="temp/detect",
            name="train_yolo11n",
            exist_ok=True,
            workers=0,
            compile=False
        )
        self.model = YOLO("temp/detect/train_yolo11n/weights/best.pt")
    
    
    def predict(self, img):
        results = self.model(
            img,
            imgsz=640,
            conf=0.6,
            iou=0.3,
            rect=True,
            half=True,
            device="0",
            augment=True
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

        results = self.model.track(
            img,
            conf=0.3,
            iou=0.2,
            rect=True,
            half=True,
            persist=True,
            tracker="botsort.yaml",
            device="0",
            augment=True
        )
        return results