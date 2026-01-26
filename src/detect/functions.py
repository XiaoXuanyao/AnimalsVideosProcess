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
import copy
from ultralytics.engine.results import Boxes, Results
from src.models.yolo import YOLOImpl
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.detect.interface import Storage


#
# =========---  1. Global key event handler   ---========= #
#


def global_key_handler(storage: Storage, config, event):
    """
    全局按键事件处理函数：处理按键，并过滤文本框输入，将所有需要按键触发的函数在此调用。
    
    :param storage: 传入interface.py的Storage实例
    :param config: 传入interface.py的Config实例
    :param event: wxPython的按键事件对象
    """
    rk = False
    rk |= label_box_idx_inc(storage, config, event)
    rk |= label_box_move(storage, config, event)
    rk |= label_box_resize(storage, config, event)
    rk |= label_box_create(storage, config, event)
    rk |= label_box_delete(storage, config, event)
    rk |= label_box_class_next(storage, config, event)
    rk |= load_result(storage, config, event)
    rk |= auto_label(storage, config, event)
    rk |= label_box_speed_reset(storage, config, event)
    rk |= label_box_speed_up(storage, config, event)
    rk |= relabel_frame(storage, config, event)
    rk |= load_last_frame_labels(storage, config, event)
    if not rk:
        event.Skip()


def global_timer_handler(storage: Storage, config):
    """
    全局定时器事件处理函数：每隔30ms执行一次，处理定时器事件，将所有需要定时触发的函数在此调用。
    
    :param storage: 传入interface.py的Storage实例
    :param config: 传入interface.py的Config实例
    """
    if storage.auto_labeling:
        frame_idx_inc(storage, config)


#
# =========---  2. Global debug output functions   ---========= #
#


def set_progress_val(storage: Storage, value):
    """
    设置全局进度条的百分比
    
    :param storage: 传入interface.py的Storage实例
    :param value: 进度条的值，范围为0到1；0表示0%，1表示100%
    """
    storage.progress_val = value
    if storage.window.outputs.progress.GetValue() == int(value * 1000):
        return
    storage.window.outputs.progress.SetValue(int(value * 1000))
    storage.window.Update()


def append_output_text(storage: Storage, text, color=(200, 200, 200), weight=wx.FONTWEIGHT_NORMAL):
    """
    在全局输出文本框中追加一行文本，并设置文本颜色和粗细。
    
    :param storage: 传入interface.py的Storage实例
    :param text: 要追加的文本内容
    :param color: 文本颜色，默认为偏白灰色(200, 200, 200)
    :param weight: 文本粗细，默认为wx.FONTWEIGHT_NORMAL
    """
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


#
# =========---  3. Calculate operations   ---========= #
#


def average(arr: list[Results], weights: list[float] | None=None) -> Boxes:
    """
    对多个Results对象的Boxes进行加权平均，返回一个新的Boxes对象。
    
    :param arr: 传入的Results对象列表
    :param weights: 权重列表，传入float数组可以计算加权平均值，默认为None，表示取平均值
    """
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


def iou_xyxy(box1, box2):
    """
    计算两个xyxy格式的边界框的IoU值。

    :param box1: 第一个边界框，格式为[x1, y1, x2, y2]，可以是numpy数组或列表
    :param box2: 第二个边界框，格式为[x1, y1, x2, y2]，可以是numpy数组或列表
    """
    box1 = np.asarray(box1)
    box2 = np.asarray(box2)

    x1 = np.maximum(box1[..., 0], box2[..., 0])
    y1 = np.maximum(box1[..., 1], box2[..., 1])
    x2 = np.minimum(box1[..., 2], box2[..., 2])
    y2 = np.minimum(box1[..., 3], box2[..., 3])

    inter_w = np.maximum(0.0, x2 - x1)
    inter_h = np.maximum(0.0, y2 - y1)
    inter = inter_w * inter_h

    area1 = np.maximum(0.0, box1[..., 2] - box1[..., 0]) * np.maximum(0.0, box1[..., 3] - box1[..., 1])
    area2 = np.maximum(0.0, box2[..., 2] - box2[..., 0]) * np.maximum(0.0, box2[..., 3] - box2[..., 1])

    union = area1 + area2 - inter
    return inter / (union + 1e-6)


def look_forward(arr: list[Results], now: Results) -> Boxes | None:
    """
    根据历史记录，返回下一帧的预测值。
    
    :param arr: 传入历史的Results对象列表
    :param now: 当前帧的Results对象
    """
    if now.boxes is None or now.boxes.data.shape[0] == 0:
        now.boxes = arr[-1].boxes
    if now.boxes is None:
        return None
    data = []
    for e in arr:
        if e.boxes is None or len(e.boxes.data) != len(now.boxes.data):
            return now.boxes
        d0 = e.boxes.data
        d0 = d0 if isinstance(d0, torch.Tensor) else torch.tensor(d0)
        d0 = d0.clone()
        order = torch.argsort(d0[:, 4])
        d0 = d0[order]
        data.append(d0.cpu().numpy())
    dx = []
    for i in range(len(data) - 1):
        dx.append(data[i + 1] - data[i])
    v = np.mean(np.array(dx), axis=0)

    dest_data = now.boxes.data
    dest_data = dest_data if isinstance(dest_data, torch.Tensor) else torch.tensor(dest_data)
    dest_data = dest_data.clone().cpu().numpy()
    pred_data = data[-1] + v
    
    iou = []
    for i in range(len(dest_data)):
        iou.append(iou_xyxy(dest_data[i][:4], pred_data[i][:4]))
    iou = np.mean(np.array(iou))

    rate = iou * 0.4 + 0.8 * 0.6
    data = pred_data * rate + dest_data * (1 - rate)
    boxes = Boxes(torch.tensor(data, dtype=torch.float32), now.boxes.orig_shape)
    return boxes


def calculate_image_results(model: YOLOImpl, frame, before_results: list[Results] | None=None) -> Results:
    """
    计算单张图像的检测框，支持传入前几帧的检测结果以进行跟踪，返回检测结果。
    
    :param model: YOLOImpl模型实例，用于计算检测框
    :param frame: 输入的图像帧
    :param before_results: 前几帧的检测结果列表，用于跟踪，默认为None
    """
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
        result: Results = model.track(frame)[0]
    else:
        result: Results = model.track(frame)[0]
    return result


def postprocess_image_results(result: Results, names: list[str]) -> Results:
    """
    对单张图像的检测结果进行后处理，主要是设置类别名称，并确保类别索引在范围内。
    
    :param result: 单张图像的检测结果Results对象
    :param names: 类别名称列表
    :return: 处理后的检测结果Results对象
    """
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


#
# =========---  4. Video operations   ---========= #
#


def load_videos(folder, storage: Storage, config):
    """
    加载指定文件夹中的视频文件，更新Storage中的视频列表和元数据。
    
    :param folder: 视频文件夹路径
    :param storage: 传入interface.py的Storage实例
    :param config: 传入interface.py的Config实例
    """
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
    """
    标注下一个视频
    
    :param storage: 传入interface.py的Storage实例
    :param config: 传入interface.py的Config实例
    """
    component = storage.window.load_video_opts
    if storage.video_idx < len(storage.video_list) - 1:
        component.vid_input.SetValue(str(storage.video_idx + 1))
        video_idx_set(storage, config)


def video_idx_dec(storage: Storage, config):
    """
    标注上一个视频
    
    :param storage: 传入interface.py的Storage实例
    :param config: 传入interface.py的Config实例
    """
    component = storage.window.load_video_opts
    if storage.video_idx > 0:
        component.vid_input.SetValue(str(storage.video_idx - 1))
        video_idx_set(storage, config)


def video_idx_set(storage: Storage, config):
    """
    读取要标注的视频索引（索引同视频名称），并加载对应视频的帧列表和元数据。
    
    :param storage: 传入interface.py的Storage实例
    :param config: 传入interface.py的Config实例
    """
    component = storage.window.load_video_opts
    idx_str = component.vid_input.GetValue()
    if idx_str.isdigit():
        idx = max(0, min(int(idx_str), len(storage.video_list) - 1))
        if idx == storage.video_idx:
            return
        storage.video_idx = idx
    component.video_name.SetValue(os.path.basename(storage.video_list[storage.video_idx]) if len(storage.video_list) > 0 else "未选择")
    component.vid_input.SetValue(str(storage.video_idx))
    if len(storage.video_list) > 0:  # 加载视频元数据并初始化标注数据
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
    """
    加载当前视频的所有帧，并更新进度条。
    
    :param storage: 传入interface.py的Storage实例
    :param config: 传入interface.py的Config实例
    """
    component = storage.window.load_video_opts
    set_progress_val(storage, 0)
    storage.window.outputs.progress.SetColor(wx.Colour(50, 90, 90), wx.Colour(50, 50, 50))
    cap = cv2.VideoCapture(storage.video_list[storage.video_idx])
    cnt = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    for i in range(cnt):
        ret, frame = cap.read()
        if not ret:
            break
        if i % config.frames_per_label == 0:   # 每隔几帧采样一帧用于标注
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
    """
    标注下一帧
    
    :param storage: 传入interface.py的Storage实例
    :param config: 传入interface.py的Config实例
    """
    component = storage.window.load_video_opts
    if storage.frame_idx < len(storage.frame_list) - 1:
        storage.frame_idx += 1
        component.fid_input.SetValue(str(storage.frame_idx))
        frame_idx_set(storage, config)


def frame_idx_dec(storage: Storage, config):
    """
    标注上一帧
    
    :param storage: 传入interface.py的Storage实例
    :param config: 传入interface.py的Config实例
    """
    component = storage.window.load_video_opts
    if storage.frame_idx > 0:
        storage.frame_idx -= 1
        component.fid_input.SetValue(str(storage.frame_idx))
        frame_idx_set(storage, config)


def frame_idx_set(storage: Storage, config):
    """
    设置当前标注帧的索引
    
    :param storage: 传入interface.py的Storage实例
    :param config: 传入interface.py的Config实例
    """
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
        set_progress_val(storage, len(storage.frame_res_list) / len(storage.frame_list))



def display_frame(storage: Storage, config):
    """
    在界面上显示当前标注帧。
    // 不要频繁调用此函数，此函数会重绘页面，较影响性能
    
    :param storage: 传入interface.py的Storage实例
    :param config: 传入interface.py的Config实例
    """
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
    """
    计算当前视频所有帧的CLIP特征向量，并调用选取训练集函数。
    
    :param storage: 传入interface.py的Storage实例
    :param config: 传入interface.py的Config实例
    """
    if storage.clip is None:
        append_output_text(storage, "加载CLIP模型...")
        from src.models.clip import CLIP
        storage.clip = CLIP()
        append_output_text(storage, "CLIP模型加载完成")
    if storage.yolo is None:
        append_output_text(storage, "加载YOLO模型...")
        if os.path.exists("./temp/detect/train_yolo/weights/best.pt"):
            storage.yolo = YOLOImpl(weights_path="./temp/detect/train_yolo/weights/best.pt")
            append_output_text(storage, "已加载自定义训练的YOLO模型")
        else:
            storage.yolo = YOLOImpl(weights_path="./models/" + config.model_name)
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
    """
    根据相似度阈值选取训练集。
    // 选取规则：从第一帧开始，依次计算与已选取帧的相似度，若最大相似度小于阈值则加入训练集
    
    :param storage: 传入interface.py的Storage实例
    :param config: 传入interface.py的Config实例
    :param threshold: 相似度阈值
    """
    ilist = []
    ilist.append(0)
    while True:
        sims = []
        for i in range(len(storage.frame_list)):
            v = 0
            for j in ilist:
                sim = storage.frame_emb_list[i] @ storage.frame_emb_list[j].T
                v = max(v, sim)
            sims.append(v)
        mini = sims.index(min(sims))
        if sims[mini] >= threshold:
            break
        delta = 20
        if mini not in ilist:
            ilist.append(mini)
        if mini + delta < len(storage.frame_list) and mini + delta not in ilist:
            ilist.append(mini + delta)
    ilist.sort()
    storage.train_idx_list = ilist
    storage.window.label_config.sim_th_calculate.set_status("[OK]")
    storage.window.label_config.sample_label_progress.SetLabel(f"0/{len(ilist)}")
    append_output_text(storage, f"相似度阈值：{threshold}，选取 {len(ilist)} 帧作为训练集")
    storage.window.outputs.progress.SetColor(wx.Colour(50, 200, 150), wx.Colour(50, 150, 150))
    storage.window.outputs.progress.SetValue(0)



def start_sample_label(storage: Storage, config):
    """
    开始样本标注模式
    
    :param storage: 传入interface.py的Storage实例
    :param config: 传入interface.py的Config实例
    """
    storage.status = "training"
    storage.train_idx = 0
    train_idx_set(storage, config)


def train_idx_dec(storage: Storage, config):
    """
    标注上一个训练集帧
    
    :param storage: 传入interface.py的Storage实例
    :param config: 传入interface.py的Config实例
    """
    storage.train_idx -= 1
    if storage.train_idx < 0:
        storage.train_idx = 0
    train_idx_set(storage, config)


def train_idx_inc(storage: Storage, config):
    """
    标注下一个训练集帧
    
    :param storage: 传入interface.py的Storage实例
    :param config: 传入interface.py的Config实例
    """
    storage.train_idx += 1
    if storage.train_idx >= len(storage.train_idx_list):
        storage.train_idx = len(storage.train_idx_list) - 1
    train_idx_set(storage, config)


def train_idx_set(storage: Storage, config):
    """
    设置当前标注的训练集帧索引，并对当前帧加载基于COCO数据集的预训练模型的检测结果
    
    :param storage: 传入interface.py的Storage实例
    :param config: 传入interface.py的Config实例
    """
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
    """
    在当前帧上绘制检测框和标签，返回绘制后的图像
    
    :param storage: 传入interface.py的Storage实例
    :param config: 传入interface.py的Config实例
    :param relabel: 是否重新计算检测结果，默认为False
    """
    # 计算结果，或从缓存记录中加载结果
    frame = storage.frame_list[storage.frame_idx].copy()
    result_buff: Results | None = storage.frame_res_list.get(storage.frame_idx, None)
    if result_buff is None or relabel:
        result = calculate_image_results(storage.yolo, frame, before_results=[e for e in [
            storage.frame_res_list.get(storage.frame_idx - i - 1, None) for i in range(0, 10)
        ] if e is not None])
        result = postprocess_image_results(result, storage.classes)

        if  storage.frame_idx - 3 >= 0:
            p_list = []
            for i in range(-3, 0):
                p_list.append(storage.frame_res_list.get(storage.frame_idx + i, None))
            if all(e is not None for e in p_list):
                boxes = look_forward(p_list, result)
                result.boxes = boxes
    else:
        result = result_buff

    # 使用ultralytics API绘制结果，并高亮当前训练框
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
    """
    对当前帧执行标注并显示
    
    :param storage: 传入interface.py的Storage实例
    :param config: 传入interface.py的Config实例
    :param relabel: 是否重新计算检测结果，默认为False
    """
    annotated_frame = plot_labeled_frame(storage, config, relabel=relabel)
    storage.frame_labeled_list[storage.frame_idx] = annotated_frame
    display_frame(storage, config)


def label_box_idx_inc(storage: Storage, config, event):
    """
    切换到下一个检测框
    
    :param storage: 传入interface.py的Storage实例
    :param config: 传入interface.py的Config实例
    :param event: wxPython的按键事件对象
    """
    if event.GetKeyCode() != ord('B'):
        return False
    result = storage.frame_res_list.get(storage.frame_idx, None)
    if result is None or result.boxes is None:
        return False
    box_count = result.boxes.data.shape[0]
    storage.train_box_idx = (storage.train_box_idx + 1) % box_count
    if len(storage.frame_res_list) == len(storage.train_idx_list):
        storage.window.label_config.sample_label_button.set_status("[OK]")
    label_frame(storage, config)
    return True


def label_box_move(storage: Storage, config, event):
    """
    移动当前选中的检测框
    
    :param storage: 传入interface.py的Storage实例
    :param config: 传入interface.py的Config实例
    :param event: wxPython的按键事件对象
    """
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
        return False
    result = storage.frame_res_list.get(storage.frame_idx, None)
    if result is None or result.boxes is None:
        return False
    data = result.boxes.data
    data = data if isinstance(data, torch.Tensor) else torch.tensor(data)
    data = data.clone()
    delta = storage.label_box_speed

    box = data[storage.train_box_idx]
    box[0] += direction[0] * delta
    box[1] += direction[1] * delta
    box[2] += direction[0] * delta
    box[3] += direction[1] * delta
    box[-2] = 0.95
    data[storage.train_box_idx] = box
    boxes = Boxes(data, result.boxes.orig_shape)
    result.boxes = boxes
    storage.frame_res_list[storage.frame_idx] = result
    label_frame(storage, config)
    return True


def label_box_resize(storage: Storage, config, event):
    """
    调整当前选中的检测框大小
    
    :param storage: 传入interface.py的Storage实例
    :param config: 传入interface.py的Config实例
    :param event: wxPython的按键事件对象
    """
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
        return False
    result = storage.frame_res_list.get(storage.frame_idx, None)
    if result is None or result.boxes is None:
        return False
    data = result.boxes.data
    data = data if isinstance(data, torch.Tensor) else torch.tensor(data)
    data = data.clone()
    delta = storage.label_box_speed
    
    box = data[storage.train_box_idx]
    box[2] += direction[0] * delta
    box[3] += direction[1] * delta
    box[-2] = 0.95
    data[storage.train_box_idx] = box
    boxes = Boxes(data, result.boxes.orig_shape)
    result.boxes = boxes
    storage.frame_res_list[storage.frame_idx] = result
    label_frame(storage, config)
    return True


def label_box_speed_up(storage: Storage, config, event):
    """
    在持续按键过程中，提高检测框移动和调整的速度
    
    :param storage: 传入interface.py的Storage实例
    :param config: 传入interface.py的Config实例
    :param event: wxPython的按键事件对象
    """
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
        return True
    return False


def label_box_speed_reset(storage: Storage, config, event):
    """
    重置检测框移动和调整的速度
    
    :param storage: 传入interface.py的Storage实例
    :param config: 传入interface.py的Config实例
    :param event: wxPython的按键事件对象
    """
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
        return True
    return False


def label_box_create(storage: Storage, config, event):
    """
    创建一个新的检测框
    
    :param storage: 传入interface.py的Storage实例
    :param config: 传入interface.py的Config实例
    :param event: wxPython的按键事件对象
    """
    if event.GetKeyCode() != ord('C'):
        return False
    result = storage.frame_res_list.get(storage.frame_idx, None)
    if result is None or result.boxes is None:
        return False
    data = result.boxes.data
    data = data if isinstance(data, torch.Tensor) else torch.tensor(data)
    data = data.clone()
    h, w = result.boxes.orig_shape
    if data.shape[1] == 6:
        new_box = torch.tensor([[w//4, h//4, w//2, h//2, 0.95, 0]], dtype=data.dtype)
    else:
        idx = max(data[:, -3]) + 1 if data.shape[0] > 0 else 1
        new_box = torch.tensor([[w//4, h//4, w//2, h//2, idx, 0.95, 0]], dtype=data.dtype)
    data = torch.cat([data, new_box.to(data.device)], dim=0)
    boxes = Boxes(data, result.boxes.orig_shape)
    result.boxes = boxes
    result.names = {i: name for i, name in enumerate(storage.classes)}
    storage.frame_res_list[storage.frame_idx] = result
    storage.train_box_idx = data.shape[0] - 1
    label_frame(storage, config)
    return True


def label_box_delete(storage: Storage, config, event):
    """
    删除当前选中的检测框
    
    :param storage: 传入interface.py的Storage实例
    :param config: 传入interface.py的Config实例
    :param event: wxPython的按键事件对象
    """
    if event.GetKeyCode() != wx.WXK_DELETE:
        return False
    result = storage.frame_res_list.get(storage.frame_idx, None)
    if result is None or result.boxes is None:
        return False
    data = result.boxes.data
    data = data if isinstance(data, torch.Tensor) else torch.tensor(data)
    data = data.clone()
    if data.shape[0] == 0:
        return True
    data = torch.cat([data[:storage.train_box_idx], data[storage.train_box_idx+1:]], dim=0)
    boxes = Boxes(data, result.boxes.orig_shape)
    result.boxes = boxes
    storage.frame_res_list[storage.frame_idx] = result
    storage.train_box_idx = max(0, storage.train_box_idx - 1)
    label_frame(storage, config)
    return True


def label_box_class_next(storage: Storage, config, event):
    """
    切换当前选中检测框的类别到下一个类别
    
    :param storage: 传入interface.py的Storage实例
    :param config: 传入interface.py的Config实例
    :param event: wxPython的按键事件对象
    """
    if event.GetKeyCode() != ord('T'):
        return False
    result = storage.frame_res_list.get(storage.frame_idx, None)
    if result is None or result.boxes is None:
        return False
    data = result.boxes.data
    data = data if isinstance(data, torch.Tensor) else torch.tensor(data)
    data = data.clone()
    if data.shape[0] == 0:
        return True
    box = data[storage.train_box_idx]
    box[-1] = (box[-1] + 1) % len(storage.classes)
    data[storage.train_box_idx] = box
    boxes = Boxes(data, result.boxes.orig_shape)
    result.boxes = boxes
    result.names = {i: name for i, name in enumerate(storage.classes)}
    storage.frame_res_list[storage.frame_idx] = result
    label_frame(storage, config)
    return True


def save_result(storage: Storage, config, idx):
    """
    保存当前帧的标注结果到临时文件夹
    // 会覆盖原有结果文件
    
    :param storage: 传入interface.py的Storage实例
    :param config: 传入interface.py的Config实例
    :param event: wxPython的按键事件对象
    """
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

    """
    从临时文件夹加载当前帧的标注结果
    // 若无结果文件则跳过
    
    :param storage: 传入interface.py的Storage实例
    :param config: 传入interface.py的Config实例
    :param event: wxPython的按键事件对象
    """
    if event is not None:
        if event.GetKeyCode() != ord('L'):
            return False
    idx = storage.frame_idx
    p = f"data/detect/temp/{os.path.basename(storage.video_list[storage.video_idx]).replace('.mp4', '')}_frame_{idx:05d}.npy"
    if not os.path.exists(p):
        return False
    result = calculate_image_results(storage.yolo, storage.frame_list[idx])
    if result is None or result.boxes is None:
        return True
    np_data = np.load(p)
    data = torch.tensor(np_data, dtype=torch.float32)
    if data.shape[0] > 0 and data.shape[1] == 6:
        ids = torch.zeros((data.shape[0], 1), dtype=data.dtype)
        data = torch.cat([data[:, :4], ids, data[:, 4:6]], dim=1)
    boxes = Boxes(data, result.boxes.orig_shape)
    result.boxes = boxes
    storage.frame_res_list[idx] = result
    label_frame(storage, config)
    return True


def sample_train(storage: Storage, config):
    """
    使用标注好的小样本训练集帧训练YOLO模型
    
    :param storage: 传入interface.py的Storage实例
    :param config: 传入interface.py的Config实例
    """
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
    model = YOLOImpl(weights_path=f"./models/{config.model_name}")
    model.train(data=f"{p}/data.yaml", epochs=120, batch=32, optimizer="auto", lr0=3e-4, default_pretrained=f"./models/{config.model_name}")
    
    storage.yolo = model
    storage.window.label_config.train_button.set_status("[OK]")
    append_output_text(storage, "训练完成")


def append_train(storage: Storage, config):
    """
    使用当前帧增量训练YOLO模型
    // 还未实现，有问题
    
    :param storage: 传入interface.py的Storage实例
    :param config: 传入interface.py的Config实例
    """
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
    
    model = YOLOImpl(weights_path=f"./models/{config.model_name}")
    model.train(data=f"{p}/data.yaml", epochs=3, batch=4, optimizer="AdamW", lr0=3e-5, pretrained="self", default_pretrained=f"./models/{config.model_name}")
    
    storage.yolo = model
    append_output_text(storage, "单帧增量训练完成")



def start_auto_detect(storage: Storage, config):
    """
    开始自动标注模式
    
    :param storage: 传入interface.py的Storage实例
    :param config: 传入interface.py的Config实例
    """
    storage.status = "labeling"
    storage.frame_idx = 0
    storage.window.load_video_opts.fid_input.SetValue(str(storage.frame_idx))
    storage.window.outputs.progress.SetColor(wx.Colour(50, 150, 200), wx.Colour(50, 200, 150))
    frame_idx_set(storage, config)


def auto_label(storage: Storage, config, event):
    """
    切换自动标注的启停状态
    
    :param storage: 传入interface.py的Storage实例
    :param config: 传入interface.py的Config实例
    """
    if event.GetKeyCode() != ord('N'):
        return False
    if not storage.auto_labeling:
        storage.auto_labeling = True
        append_output_text(storage, "自动标注开始，按 N 键停止")
    else:
        storage.auto_labeling = False
        append_output_text(storage, "自动标注结束")
    return True


def relabel_frame(storage: Storage, config, event):
    """
    重新使用模型自动标注当前帧
    
    :param storage: 传入interface.py的Storage实例
    :param config: 传入interface.py的Config实例
    :param event: wxPython的按键事件对象
    """
    if event.GetKeyCode() != ord('R') or storage.status != "labeling":
        return False
    label_frame(storage, config, relabel=True)
    return True


def load_last_frame_labels(storage: Storage, config, event):
    """
    复制上一帧的标注到当前帧
    
    :param storage: 传入interface.py的Storage实例
    :param config: 传入interface.py的Config实例
    :param event: wxPython的按键事件对象
    """
    if event.GetKeyCode() != ord('P'):
        return False
    result: Results | None = None
    if storage.status == "labeling":
        if storage.frame_res_list.get(storage.frame_idx - 1, None) is not None:
            result = storage.frame_res_list[storage.frame_idx - 1]
    if storage.status == "training":
        if storage.frame_res_list.get(storage.train_idx_list[storage.train_idx - 1], None) is not None:
            result = storage.frame_res_list[storage.train_idx_list[storage.train_idx - 1]]
    
    # 复制结果，使用深拷贝避免引用问题
    if result is None:
        return True
    result = copy.deepcopy(result)
    if result.boxes is not None:
        data = result.boxes.data
        data = data if isinstance(data, torch.Tensor) else torch.tensor(data)
        data = data.clone()
        boxes = Boxes(data, result.boxes.orig_shape)
        result.boxes = boxes
    
    if storage.status == "labeling":
        storage.frame_res_list[storage.frame_idx] = result
    if storage.status == "training":
        storage.frame_res_list[storage.train_idx_list[storage.train_idx]] = result
    label_frame(storage, config)
    return True


def save_results(storage: Storage, config):
    """
    保存当前选中视频的所有标注结果到data/detect目录下，包括JSON标签文件和标注预览视频文件。
    
    :param storage: 传入interface.py的Storage实例
    :param config: 传入interface.py的Config实例
    """
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
            final_output[f"frame_{(i * config.frames_per_label):05d}"] = labels
        json.dump(final_output, f, indent=4)
    append_output_text(storage, f"已保存检测结果至 data/detect/labels/{name}.json")

    vwr = cv2.VideoWriter(
        f"data/detect/videos/{name}.mp4",
        cv2.VideoWriter_fourcc(*'mp4v'),  # type: ignore
        30 / config.frames_per_label,
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