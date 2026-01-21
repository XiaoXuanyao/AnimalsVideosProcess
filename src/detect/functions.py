from __future__ import annotations

import os
import cv2
import shutil
import wx
import threading
import time
import torch
import csv
import json
import numpy as np
from ultralytics.engine.results import Boxes, Results
from src.models.yolo import YOLOImpl
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.detect.interface import Storage


# =========---  1. Global key event handler   ---========= #


def global_key_handler(storage: Storage, config, event):
    label_box_idx_inc(storage, config, event)
    label_box_move(storage, config, event)
    label_box_resize(storage, config, event)
    label_box_create(storage, config, event)
    label_box_delete(storage, config, event)
    label_box_class_next(storage, config, event)
    load_result(storage, config, event)
    auto_label(storage, config, event)
    label_box_speed_reset(storage, config, event)
    label_box_speed_up(storage, config, event)
    relabel_frame(storage, config, event)
    load_last_frame_labels(storage, config, event)
    event.Skip()

def global_timer_handler(storage: Storage, config):
    if storage.auto_labeling:
        frame_idx_inc(storage, config)


# =========---  2. Global debug output functions   ---========= #


def set_progress_val(storage: Storage, value):
    storage.progress_val = value
    if storage.window.outputs.progress.GetValue() == int(value * 1000):
        return
    storage.window.outputs.progress.SetValue(int(value * 1000))
    storage.window.Update()

def append_output_text(storage: Storage, text, color=(200, 200, 200), weight=wx.FONTWEIGHT_NORMAL):
    start = storage.window.outputs.output_field.GetLastPosition()
    if storage.window.outputs.output_field.GetLastPosition() > 0:
        storage.window.outputs.output_field.AppendText("\n")
    storage.window.outputs.output_field.AppendText(time.strftime("[%H:%M:%S] ", time.localtime()))
    storage.window.outputs.output_field.AppendText(text)
    end = storage.window.outputs.output_field.GetLastPosition()
    attr = wx.TextAttr()
    attr.SetTextColour(wx.Colour(*color))
    attr.SetFontWeight(weight)
    attr.SetLineSpacing(8)
    attr.SetParagraphSpacingBefore(2)
    attr.SetParagraphSpacingAfter(2)
    storage.window.outputs.output_field.SetStyle(start, end, attr)
    storage.window.outputs.output_field.ShowPosition(end)
    storage.window.Update()


# =========---  3. Calculate operations   ---========= #


def average(arr: list[Results], weights: list[float] | None=None) -> Boxes:
    len_arr = [0 if e.boxes is None else len(e.boxes.data) for e in arr]
    len_map = {}
    for i, e in enumerate(len_arr):
        if e not in len_map:
            len_map[e] = 0
        len_map[e] += weights[i] if weights is not None else 1
    std_len = max(len_map.items(), key=lambda x: x[1])[0]
    use_idx = [i for i, e in enumerate(len_arr) if e == std_len]
    data = []
    for e in arr:
        if e.boxes is None or len(e.boxes.data) != std_len:
            continue
        boxes_data = e.boxes.data
        boxes_data = boxes_data if isinstance(boxes_data, torch.Tensor) else torch.tensor(boxes_data)
        boxes_data = boxes_data.clone()
        data.append(boxes_data.cpu().numpy())
    data = np.array(data)
    avg_data = np.average(data, axis=0, weights=[weights[i] for i in use_idx] if weights is not None else None)
    orig_shape = [e.boxes.orig_shape for e in arr if e.boxes is not None and len(e.boxes.data) == std_len][0]
    boxes = Boxes(torch.tensor(avg_data, dtype=torch.float32), orig_shape)
    return boxes

def calculate_image_results(model: YOLOImpl, frame, before_results: list[Results] | None=None) -> Results:
    if before_results is not None and len(before_results) > 0:
        detects = []
        for e in before_results:
            if e is None:
                break
            boxes = e.boxes
            if boxes is None:
                continue
            xyxy = boxes.xyxy
            conf = boxes.conf
            cls = boxes.cls
            if isinstance(xyxy, torch.Tensor):
                xyxy = xyxy.cpu().numpy()
            if isinstance(conf, torch.Tensor):
                conf = conf.cpu().numpy()
            if isinstance(cls, torch.Tensor):
                cls = cls.cpu().numpy()
            xyxy = np.array(xyxy)
            conf = np.array(conf)
            cls = np.array(cls)
            conf = conf.reshape(-1, 1)
            cls = cls.reshape(-1, 1)
            detects.append(np.concatenate([xyxy, conf, cls], axis=1).astype(np.float32))
        result: Results = model.track(
            img=frame,
            detects=detects
        )[0]
    else:
        result: Results = model.predict(frame)[0]
    return result

def postprocess_image_results(result: Results, names: list[str]) -> Results:
    result.names = {i: name for i, name in enumerate(names)}
    if result.boxes is not None:
        data = result.boxes.data
        data = data if isinstance(data, torch.Tensor) else torch.tensor(data)
        data = data.clone()
        for i in range(data.shape[0]):
            data[i][-1] = min(data[i][-1], len(names) - 1)
        boxes = Boxes(data, result.boxes.orig_shape)
        result.boxes = boxes
    return result


# =========---  4. Video operations   ---========= #


def load_videos(folder, storage: Storage, config):
    component = storage.window.load_video_opts
    vlist = []
    for e in os.listdir(folder):
        if e.endswith(('.mp4', '.avi', '.mov', '.mkv')):
            p = os.path.join(folder, e)
            vlist.append(p)
    storage.video_list = vlist
    component.path_count.SetLabel(str(len(vlist)))
    with open(component.metadata_input.GetValue(), 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            storage.metadata[row["name"]] = row
    append_output_text(storage, f"已加载 {len(vlist)} 个视频文件")
    video_idx_set(storage, config)

def video_idx_inc(storage: Storage, config):
    component = storage.window.load_video_opts
    if storage.video_idx < len(storage.video_list) - 1:
        storage.video_idx += 1
        component.vid_input.SetValue(str(storage.video_idx))
        video_idx_set(storage, config)

def video_idx_dec(storage: Storage, config):
    component = storage.window.load_video_opts
    if storage.video_idx > 0:
        storage.video_idx -= 1
        component.vid_input.SetValue(str(storage.video_idx))
        video_idx_set(storage, config)

def video_idx_set(storage: Storage, config):
    component = storage.window.load_video_opts
    idx_str = component.vid_input.GetValue()
    if idx_str.isdigit():
        idx = max(0, min(int(idx_str), len(storage.video_list) - 1))
        storage.video_idx = idx
    component.video_name.SetValue(os.path.basename(storage.video_list[storage.video_idx]) if len(storage.video_list) > 0 else "未选择")
    component.vid_input.SetValue(str(storage.video_idx))
    if len(storage.video_list) > 0:
        video_name = os.path.basename(storage.video_list[storage.video_idx])
        classes = storage.metadata.get(video_name, {}).get("title", "")
        classes = classes.split(" ")[1].split("/")
        storage.frame_list = []
        storage.frame_emb_list = []
        storage.frame_idx = 0
        storage.frame_labeled_list = {}
        storage.frame_res_list = {}
        storage.train_idx_list = []
        storage.train_idx = 0
        storage.train_box_idx = 0
        storage.classes = classes
    if len(storage.video_list) > 0:
        load_frames(storage, config)


def load_frames(storage: Storage, config):
    component = storage.window.load_video_opts
    set_progress_val(storage, 0)
    storage.window.outputs.progress.SetColor(wx.Colour(50, 90, 90), wx.Colour(50, 50, 50))
    cap = cv2.VideoCapture(storage.video_list[storage.video_idx])
    cnt = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    for i in range(cnt):
        ret, frame = cap.read()
        if not ret:
            break
        storage.frame_list.append(frame)
        if i % 10 == 0 or i == cnt:
            set_progress_val(storage, i / cnt)
    cap.release()
    storage.window.load_video_opts.load_button.set_status("[OK]")
    storage.frame_idx = 0
    component.fid_input.SetValue(str(storage.frame_idx))
    append_output_text(storage, f"已加载视频{os.path.basename(storage.video_list[storage.video_idx])}，共 {len(storage.frame_list)} 帧")
    frame_idx_set(storage, config)

def frame_idx_inc(storage: Storage, config):
    component = storage.window.load_video_opts
    if storage.frame_idx < len(storage.frame_list) - 1:
        storage.frame_idx += 1
        component.fid_input.SetValue(str(storage.frame_idx))
        frame_idx_set(storage, config)

def frame_idx_dec(storage: Storage, config):
    component = storage.window.load_video_opts
    if storage.frame_idx > 0:
        storage.frame_idx -= 1
        component.fid_input.SetValue(str(storage.frame_idx))
        frame_idx_set(storage, config)

def frame_idx_set(storage: Storage, config):
    component = storage.window.load_video_opts
    idx_str = component.fid_input.GetValue()
    if idx_str.isdigit():
        idx = max(0, min(int(idx_str), len(storage.frame_list) - 1))
        storage.frame_idx = idx
    component.fid_input.SetValue(str(storage.frame_idx))
    component.frame_ord.SetLabel(f"{storage.frame_idx}/{len(storage.frame_list)-1}")
    display_frame(storage, config)
    if storage.status == "labeling":
        label_frame(storage, config)


def display_frame(storage: Storage, config):
    if len(storage.frame_list) > 0:
        frame = storage.frame_list[storage.frame_idx] if storage.frame_labeled_list.get(storage.frame_idx) is None else storage.frame_labeled_list[storage.frame_idx]
        frame = cv2.resize(frame, config.frame_size)
        height, width = frame.shape[:2]
        bmp = wx.Bitmap.FromBuffer(width, height, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        comp = storage.window.frame_display.frame
        storage.window.frame_display.Freeze()
        comp.SetBitmap(wx.BitmapBundle(bmp))
        storage.window.frame_display.Thaw()
        if storage.frame_res_list.get(storage.frame_idx, None) is not None:
            storage.window.Update()


def get_clip_features(storage: Storage, config):
    if storage.clip is None:
        append_output_text(storage, "加载CLIP模型...")
        from src.models.clip import CLIP
        storage.clip = CLIP()
        append_output_text(storage, "CLIP模型加载完成")
    if storage.yolo is None:
        append_output_text(storage, "加载YOLO模型...")
        if os.path.exists("temp/detect/train_yolo/weights/best.pt"):
            storage.yolo = YOLOImpl(weights_path="temp/detect/train_yolo/weights/best.pt")
            append_output_text(storage, "已加载自定义训练的YOLO模型")
        else:
            storage.yolo = YOLOImpl(weights_path="models/" + config.model_name)
            append_output_text(storage, "YOLO模型加载完成")
    storage.window.outputs.progress.SetColor(wx.Colour(50, 150, 150), wx.Colour(50, 90, 90))
    storage.window.outputs.progress.SetValue(0)
    storage.frame_emb_list = []
    for i in range(len(storage.frame_list)):
        frame = storage.frame_list[i]
        frame = cv2.resize(frame, (224, 224))
        emb = storage.clip.calculate_image(frame)
        storage.frame_emb_list.append(emb)
        storage.window.outputs.progress.SetValue(int((i + 1) / len(storage.frame_list) * 1000))
        storage.window.Update()
    append_output_text(storage, "相似度特征计算完成")
    threshold = float(storage.window.label_config.sim_th_input.GetValue())
    get_train_set(storage, config, threshold=threshold)

def get_train_set(storage: Storage, config, threshold):
    ilist = []
    ilist.append(0)
    for i in range(1, len(storage.frame_emb_list)):
        sims = []
        for j in ilist:
            sim = storage.frame_emb_list[i] @ storage.frame_emb_list[j].T
            sims.append(sim)
        max_sim = max(sims)
        if max_sim < threshold:
            ilist.append(i)
    storage.train_idx_list = ilist
    storage.window.label_config.sim_th_calculate.set_status("[OK]")
    storage.window.label_config.sample_label_progress.SetLabel(f"0/{len(ilist)}")
    append_output_text(storage, f"相似度阈值：{threshold}，选取 {len(ilist)} 帧作为训练集")
    storage.window.outputs.progress.SetColor(wx.Colour(50, 200, 100), wx.Colour(50, 150, 150))
    storage.window.outputs.progress.SetValue(0)


def start_sample_label(storage: Storage, config):
    storage.status = "training"
    storage.train_idx = 0
    train_idx_set(storage, config)

def train_idx_dec(storage: Storage, config):
    storage.train_idx -= 1
    if storage.train_idx < 0:
        storage.train_idx = 0
    train_idx_set(storage, config)

def train_idx_inc(storage: Storage, config):
    storage.train_idx += 1
    if storage.train_idx >= len(storage.train_idx_list):
        storage.train_idx = len(storage.train_idx_list) - 1
    train_idx_set(storage, config)

def train_idx_set(storage: Storage, config):
    save_result(storage, config, storage.frame_idx)
    idx = storage.train_idx_list[storage.train_idx]
    storage.frame_idx = idx
    storage.window.load_video_opts.fid_input.SetValue(str(storage.frame_idx))
    storage.window.label_config.sample_label_progress.SetLabel(f"{storage.train_idx}/{len(storage.train_idx_list)}")
    frame_idx_set(storage, config)
    load_result(storage, config, None)
    label_frame(storage, config)
    set_progress_val(storage, len(storage.frame_res_list) / len(storage.train_idx_list))


def plot_labeled_frame(storage: Storage, config, relabel=False):
    frame = storage.frame_list[storage.frame_idx].copy()
    result_buff: Results | None = storage.frame_res_list.get(storage.frame_idx, None)
    if result_buff is None or relabel:
        result = calculate_image_results(storage.yolo, frame, before_results=[e for e in [
            storage.frame_res_list.get(storage.frame_idx - i - 1, None) for i in range(0, 10)
        ] if e is not None])
        result = postprocess_image_results(result, storage.classes)

        if  storage.frame_idx - 1 >= 0 and storage.frame_idx + 1 < len(storage.frame_list):
            a_results = []
            for i in range(-1, 2):
                frame_i = storage.frame_idx + i
                res_i = storage.frame_res_list.get(frame_i, None)
                if res_i is None:
                    res_i = calculate_image_results(storage.yolo, storage.frame_list[frame_i], before_results=[e for e in [
                        storage.frame_res_list.get(frame_i - j - 1, None) for j in range(0, 10)
                    ] if e is not None])
                    res_i = postprocess_image_results(res_i, storage.classes)
            boxes = average([result] + a_results, weights=[0.4] + [0.6 / len(a_results) for _ in a_results])
            result.boxes = boxes
    else:
        result = result_buff

    result.names = {i: name for i, name in enumerate(storage.classes)}
    storage.frame_res_list[storage.frame_idx] = result
    if result.boxes is None:
        return frame
    data = result.boxes.data
    data = data if isinstance(data, torch.Tensor) else torch.tensor(data)
    data = data.clone()
    origin_boxes = result.boxes
    for i in range(data.shape[0]):
        data0 = data[i].reshape(1, -1)
        boxes = Boxes(data0, result.boxes.orig_shape)
        result.boxes = boxes
        result.names = {i: name for i, name in enumerate(storage.classes)}
        line_weight = 2
        frame = result.plot(
            img=frame,
            labels=True,
            boxes=True,
            masks=True,
            probs=True,
            line_width=line_weight
        )
        if storage.train_box_idx == i:
            line_weight = 4
        frame = result.plot(
            img=frame,
            labels=False,
            boxes=True,
            masks=False,
            probs=False,
            line_width=line_weight
        )
    result.boxes = origin_boxes
    return frame

def label_frame(storage: Storage, config, relabel=False):
    annotated_frame = plot_labeled_frame(storage, config, relabel=relabel)
    storage.frame_labeled_list[storage.frame_idx] = annotated_frame
    display_frame(storage, config)

def label_box_idx_inc(storage: Storage, config, event):
    if event.GetKeyCode() != ord('B'):
        return
    result = storage.frame_res_list.get(storage.frame_idx, None)
    if result is None or result.boxes is None:
        return
    box_count = result.boxes.data.shape[0]
    storage.train_box_idx = (storage.train_box_idx + 1) % box_count
    if len(storage.frame_res_list) == len(storage.train_idx_list):
        storage.window.label_config.sample_label_button.set_status("[OK]")
    label_frame(storage, config)

def label_box_move(storage: Storage, config, event):
    direction = (0, 0)
    if event.GetKeyCode() == wx.WXK_UP:
        direction = (0, -1)
    elif event.GetKeyCode() == wx.WXK_DOWN:
        direction = (0, 1)
    elif event.GetKeyCode() == wx.WXK_LEFT:
        direction = (-1, 0)
    elif event.GetKeyCode() == wx.WXK_RIGHT:
        direction = (1, 0)
    else:
        return
    result = storage.frame_res_list.get(storage.frame_idx, None)
    if result is None or result.boxes is None:
        return
    data = result.boxes.data
    data = data if isinstance(data, torch.Tensor) else torch.tensor(data)
    data = data.clone()
    delta = storage.label_box_speed

    box = data[storage.train_box_idx]
    box[0] += direction[0] * delta
    box[1] += direction[1] * delta
    box[2] += direction[0] * delta
    box[3] += direction[1] * delta
    data[storage.train_box_idx] = box
    boxes = Boxes(data, result.boxes.orig_shape)
    result.boxes = boxes
    storage.frame_res_list[storage.frame_idx] = result
    label_frame(storage, config)

def label_box_resize(storage: Storage, config, event):
    direction = (0, 0)
    if event.GetKeyCode() == ord('W'):
        direction = (0, -1)
    elif event.GetKeyCode() == ord('S'):
        direction = (0, 1)
    elif event.GetKeyCode() == ord('A'):
        direction = (-1, 0)
    elif event.GetKeyCode() == ord('D'):
        direction = (1, 0)
    else:
        return
    result = storage.frame_res_list.get(storage.frame_idx, None)
    if result is None or result.boxes is None:
        return
    data = result.boxes.data
    data = data if isinstance(data, torch.Tensor) else torch.tensor(data)
    data = data.clone()
    delta = storage.label_box_speed
    
    box = data[storage.train_box_idx]
    box[2] += direction[0] * delta
    box[3] += direction[1] * delta
    data[storage.train_box_idx] = box
    boxes = Boxes(data, result.boxes.orig_shape)
    result.boxes = boxes
    storage.frame_res_list[storage.frame_idx] = result
    label_frame(storage, config)

def label_box_speed_up(storage: Storage, config, event):
    if event.GetKeyCode() in [
        ord('W'),
        ord('A'),
        ord('S'),
        ord('D'),
        wx.WXK_UP,
        wx.WXK_DOWN,
        wx.WXK_LEFT,
        wx.WXK_RIGHT
    ]:
        storage.label_box_speed = storage.label_box_speed * 1.02 + 0.03
        storage.last_speed_up_time = time.time()

def label_box_speed_reset(storage: Storage, config, event):
    if storage.last_speed_up_time + 0.1 < time.time() and event.GetKeyCode() in [
        ord('W'),
        ord('A'),
        ord('S'),
        ord('D'),
        wx.WXK_UP,
        wx.WXK_DOWN,
        wx.WXK_LEFT,
        wx.WXK_RIGHT
    ]:
        storage.label_box_speed = 1.0

def label_box_create(storage: Storage, config, event):
    if event.GetKeyCode() != ord('C'):
        return
    result = storage.frame_res_list.get(storage.frame_idx, None)
    if result is None or result.boxes is None:
        return
    data = result.boxes.data
    data = data if isinstance(data, torch.Tensor) else torch.tensor(data)
    data = data.clone()
    h, w = result.boxes.orig_shape
    if data.shape[1] == 6:
        new_box = torch.tensor([[w//4, h//4, w//2, h//2, 0.9, 0]], dtype=data.dtype)
    else:
        new_box = torch.tensor([[w//4, h//4, w//2, h//2, max(data[:, -3]) + 1, 0.9, 0]], dtype=data.dtype)
    data = torch.cat([data, new_box.to(data.device)], dim=0)
    boxes = Boxes(data, result.boxes.orig_shape)
    result.boxes = boxes
    result.names = {i: name for i, name in enumerate(storage.classes)}
    storage.frame_res_list[storage.frame_idx] = result
    storage.train_box_idx = data.shape[0] - 1
    label_frame(storage, config)

def label_box_delete(storage: Storage, config, event):
    if event.GetKeyCode() != wx.WXK_DELETE:
        return
    result = storage.frame_res_list.get(storage.frame_idx, None)
    if result is None or result.boxes is None:
        return
    data = result.boxes.data
    data = data if isinstance(data, torch.Tensor) else torch.tensor(data)
    data = data.clone()
    if data.shape[0] == 0:
        return
    data = torch.cat([data[:storage.train_box_idx], data[storage.train_box_idx+1:]], dim=0)
    boxes = Boxes(data, result.boxes.orig_shape)
    result.boxes = boxes
    storage.frame_res_list[storage.frame_idx] = result
    storage.train_box_idx = max(0, storage.train_box_idx - 1)
    label_frame(storage, config)

def label_box_class_next(storage: Storage, config, event):
    if event.GetKeyCode() != ord('T'):
        return
    result = storage.frame_res_list.get(storage.frame_idx, None)
    if result is None or result.boxes is None:
        return
    data = result.boxes.data
    data = data if isinstance(data, torch.Tensor) else torch.tensor(data)
    data = data.clone()
    if data.shape[0] == 0:
        return
    box = data[storage.train_box_idx]
    box[-1] = (box[-1] + 1) % len(storage.classes)
    data[storage.train_box_idx] = box
    boxes = Boxes(data, result.boxes.orig_shape)
    result.boxes = boxes
    result.names = {i: name for i, name in enumerate(storage.classes)}
    storage.frame_res_list[storage.frame_idx] = result
    label_frame(storage, config)

def save_result(storage: Storage, config, idx):
    p = f"data/detect/temp/{os.path.basename(storage.video_list[storage.video_idx]).replace('.mp4', '')}_frame_{idx:05d}.npy"
    os.makedirs(os.path.dirname(p), exist_ok=True)
    result = storage.frame_res_list.get(idx, None)
    if result is None or result.boxes is None:
        return
    data = result.boxes.data
    data = data if isinstance(data, torch.Tensor) else torch.tensor(data)
    data = data.clone()
    np_data = data.cpu().numpy()
    np.save(p, np_data)

def load_result(storage: Storage, config, event):
    if event is not None:
        if event.GetKeyCode() != ord('L'):
            return
    idx = storage.frame_idx
    p = f"data/detect/temp/{os.path.basename(storage.video_list[storage.video_idx]).replace('.mp4', '')}_frame_{idx:05d}.npy"
    if not os.path.exists(p):
        return
    result = calculate_image_results(storage.yolo, storage.frame_list[idx])
    if result is None or result.boxes is None:
        return
    np_data = np.load(p)
    data = torch.tensor(np_data, dtype=torch.float32)
    boxes = Boxes(data, result.boxes.orig_shape)
    result.boxes = boxes
    storage.frame_res_list[idx] = result
    label_frame(storage, config)

def sample_train(storage: Storage, config):
    storage.window.label_config.sample_label_progress.SetLabel(f"{len(storage.train_idx_list)}/{len(storage.train_idx_list)}")
    storage.window.label_config.sample_label_button.set_status("[OK]")
    append_output_text(storage, "创建训练集目录，路径：temp/dataset")
    p = "temp/dataset"
    if os.path.exists(p):
        shutil.rmtree(p)
    if os.path.exists("temp/detect/train_yolo"):
        shutil.rmtree("temp/detect/train_yolo")
    os.makedirs(f"{p}/images/train", exist_ok=True)
    os.makedirs(f"{p}/images/val", exist_ok=True)
    os.makedirs(f"{p}/labels/train", exist_ok=True)
    os.makedirs(f"{p}/labels/val", exist_ok=True)
    yaml = {
        "train": "./",
        "val": "./",
        "nc": len(storage.classes),
        "names": storage.classes
    }
    json.dump(yaml, open(f"{p}/data.yaml", 'w'), indent=4)

    append_output_text(storage, "写入标注数据...")
    for idx in storage.train_idx_list:
        result = storage.frame_res_list.get(idx, None)
        if result is None:
            continue
        img_path = f"{p}/images/train/frame_{idx:05d}.jpg"
        lab_path = f"{p}/labels/train/frame_{idx:05d}.txt"
        cv2.imwrite(img_path, storage.frame_list[idx])
        with open(lab_path, 'w') as f:
            if result.boxes is None:
                continue
            data = result.boxes.data
            data = data if isinstance(data, torch.Tensor) else torch.tensor(data)
            data = data.clone()
            h, w = result.boxes.orig_shape
            for i in range(data.shape[0]):
                box = data[i]
                x_center = (box[0] + box[2]) / 2 / w
                y_center = (box[1] + box[3]) / 2 / h
                box_w = (box[2] - box[0]) / w
                box_h = (box[3] - box[1]) / h
                cls = int(box[-1])
                f.write(f"{cls} {x_center:.4f} {y_center:.4f} {box_w:.4f} {box_h:.4f}\n")
    
    append_output_text(storage, "训练模型...")
    model = YOLOImpl(weights_path=f"models/{config.model_name}")
    model.train(data=f"{p}/data.yaml", epochs=120, batch=32, optimizer="auto", lr=3e-4)
    
    storage.yolo = model
    storage.window.label_config.train_button.set_status("[OK]")
    append_output_text(storage, "训练完成")

def append_train(storage: Storage, config):
    p = "temp/dataset"
    if os.path.exists(p):
        shutil.rmtree(p)
    os.makedirs(f"{p}/images/train", exist_ok=True)
    os.makedirs(f"{p}/images/val", exist_ok=True)
    os.makedirs(f"{p}/labels/train", exist_ok=True)
    os.makedirs(f"{p}/labels/val", exist_ok=True)

    yaml = {
        "train": "./",
        "val": "./",
        "nc": len(storage.classes),
        "names": storage.classes
    }
    json.dump(yaml, open(f"{p}/data.yaml", 'w'), indent=4)

    append_output_text(storage, "写入标注数据...")
    for idx in storage.train_idx_list + [storage.frame_idx]:
        result = storage.frame_res_list.get(idx, None)
        if result is None:
            continue
        img_path = f"{p}/images/train/frame_{idx:05d}.jpg"
        lab_path = f"{p}/labels/train/frame_{idx:05d}.txt"
        cv2.imwrite(img_path, storage.frame_list[idx])
        with open(lab_path, 'w') as f:
            if result.boxes is None:
                continue
            data = result.boxes.data
            data = data if isinstance(data, torch.Tensor) else torch.tensor(data)
            data = data.clone()
            h, w = result.boxes.orig_shape
            for i in range(data.shape[0]):
                box = data[i]
                x_center = (box[0] + box[2]) / 2 / w
                y_center = (box[1] + box[3]) / 2 / h
                box_w = (box[2] - box[0]) / w
                box_h = (box[3] - box[1]) / h
                cls = int(box[-1])
                f.write(f"{cls} {x_center:.4f} {y_center:.4f} {box_w:.4f} {box_h:.4f}\n")
    
    model = YOLOImpl(weights_path=f"models/{config.model_name}")
    model.train(data=f"{p}/data.yaml", epochs=3, batch=4, optimizer="Adam", lr=1e-4, pretrained="self")
    
    storage.yolo = model
    append_output_text(storage, "单帧增量训练完成")


def start_auto_detect(storage: Storage, config):
    storage.status = "labeling"
    storage.frame_idx = 0
    storage.window.load_video_opts.fid_input.SetValue(str(storage.frame_idx))
    frame_idx_set(storage, config)

def auto_label(storage: Storage, config, event):
    if event.GetKeyCode() != ord('N'):
        return
    if not storage.auto_labeling:
        storage.auto_labeling = True
        append_output_text(storage, "自动标注开始，按 N 键停止")
    else:
        storage.auto_labeling = False
        append_output_text(storage, "自动标注结束")

def relabel_frame(storage: Storage, config, event):
    if event.GetKeyCode() != ord('R') or storage.status != "labeling":
        return
    label_frame(storage, config, relabel=True)

def load_last_frame_labels(storage: Storage, config, event):
    if event.GetKeyCode() != ord('P'):
        return
    if storage.status == "labeling":
        if storage.frame_res_list.get(storage.frame_idx - 1, None) is not None:
            storage.frame_res_list[storage.frame_idx] = storage.frame_res_list[storage.frame_idx - 1]
    if storage.status == "training":
        if storage.frame_res_list.get(storage.train_idx_list[storage.train_idx - 1], None) is not None:
            storage.frame_res_list[storage.train_idx_list[storage.train_idx]] = storage.frame_res_list[storage.train_idx_list[storage.train_idx - 1]]
    label_frame(storage, config)

def save_results(storage: Storage, config):
    name = os.path.basename(storage.video_list[storage.video_idx].replace(".mp4", ""))
    os.makedirs("data/detect/labels", exist_ok=True)
    os.makedirs("data/detect/videos", exist_ok=True)
    with open(f"data/detect/labels/{name}.json", "w", encoding="utf-8") as f:
        final_output = {}
        for i in range(0, len(storage.frame_list)):
            result = storage.frame_res_list.get(i, None)
            if result is None or result.boxes is None:
                continue
            labels = {
                "xywhn": result.boxes.xywhn.tolist(),
                "conf": result.boxes.conf.tolist(),
                "cls": result.boxes.cls.tolist(),
                "names": [result.names[int(c)] for c in result.boxes.cls],
            }
            final_output[f"frame_{i:05d}"] = labels
        json.dump(final_output, f, indent=4)
    append_output_text(storage, f"已保存检测结果至 data/detect/labels/{name}.json")

    vwr = cv2.VideoWriter(
        f"data/detect/videos/{name}.mp4",
        cv2.VideoWriter_fourcc(*'mp4v'),  # type: ignore
        30,
        config.frame_size
    )
    for i in range(0, len(storage.frame_list)):
        frame = storage.frame_list[i]
        result = storage.frame_res_list.get(i, None)
        annotated_frame = result.plot(img=frame, line_width=2) if result is not None else frame
        annotated_frame = cv2.resize(annotated_frame, config.frame_size)
        vwr.write(annotated_frame)
    vwr.release()
    append_output_text(storage, f"已保存检测视频至 data/detect/videos/{name}.mp4")