from __future__ import annotations

if __name__ == "__main__":

    import wx
    from wx.lib.buttons import GenButton
    from wx.richtext import RichTextCtrl
    from src.display.display_lib import *
    from src.detect.functions import *

    from typing import TYPE_CHECKING
    if TYPE_CHECKING:
        from src.models.clip import CLIP
        from src.models.yolo import YOLOImpl


    class Storage():
        """
        全局存储类，存储程序运行时的各种状态和数据
        // 变量类型注释使用了`from __future__ import annotations`以避免循环
        """
        def __init__(self):
            self.clip: CLIP = None  # type: ignore
            self.yolo: YOLOImpl = None  # type: ignore

            self.window: DetectFrame

            self.metadata: dict[str, dict] = {}
            self.video_list: list[str] = []
            self.video_idx = -1
            self.frame_list = []
            self.frame_emb_list = []
            self.frame_idx = 0
            self.frame_labeled_list = {}
            self.frame_res_list: dict[int, Results] = {}

            self.train_idx_list: list[int] = []
            self.train_idx = 0
            self.train_box_idx = 0
            self.classes: list[str] = []
            self.label_box_speed = 1.0
            self.last_speed_up_time = 0.0

            self.progress_val = 0
            self.status = "preparing"
            self.auto_labeling = False
    
    class Config():
        """
        全局配置类，存储程序运行时的各种配置参数
        """
        def __init__(self):
            self.version = "v0.0.1"
            self.model_name = "yolo11n.pt"
            self.frame_size = (640, 360)


    #
    # =========---  Main GUI Frame   ---========= #
    #


    app = wx.App(False)
    storage = Storage()
    config = Config()


    class LoadVideoOpts(Panel):
        """
        视频加载与选择面板
        """
        def __init__(self, parent):
            super().__init__(parent)
            ar = wx.ALIGN_RIGHT | wx.ALIGN_CENTER_VERTICAL
            # 视频文件夹选择
            self.metadata_title = wx.StaticText(self, label="元数据文件路径")
            self.metadata_input = wx.TextCtrl(self, value="data/metadata/metadata.csv", size=wx.Size(160, -1))
            self.path_title = wx.StaticText(self, label="视频文件夹路径")
            self.path_input = wx.TextCtrl(self, value="data/truncated_videos", size=wx.Size(160, -1))
            self.path_count = wx.StaticText(self, label="0", size=wx.Size(30, -1), style=ar)
            self.load_button = ButtonWithStatus(self, label="加载视频")
            # 视频选择
            self.vid_title = wx.StaticText(self, label="视频索引")
            self.video_name = wx.TextCtrl(self, value="未选择", size=wx.Size(160, -1), style=wx.TE_READONLY)
            self.vid_input = wx.TextCtrl(self, value="0", size=wx.Size(60, -1))
            self.vid_modifier = Panel(self, vgap=0, hgap=0)
            self.vid_inc = GenButton(self.vid_modifier, label="+", size=wx.Size(33, -1))
            self.vid_dec = GenButton(self.vid_modifier, label="-", size=wx.Size(33, -1))
            # 帧选择
            self.fid_title = wx.StaticText(self, label="帧索引")
            self.frame_ord = wx.StaticText(self, label="当前帧/总帧数", size=wx.Size(120, -1), style=ar)
            self.fid_input = wx.TextCtrl(self, value="0", size=wx.Size(60, -1))
            self.fid_modifier = Panel(self, vgap=0, hgap=0)
            self.fid_inc = GenButton(self.fid_modifier, label="+", size=wx.Size(33, -1))
            self.fid_dec = GenButton(self.fid_modifier, label="-", size=wx.Size(33, -1))

            lc = wx.ALIGN_CENTER_VERTICAL | wx.ALIGN_LEFT
            rc = wx.ALIGN_CENTER_VERTICAL | wx.ALIGN_RIGHT
            self.add(self.metadata_title, pos=(0, 0), flag=rc)
            self.add(self.metadata_input, pos=(0, 1), flag=rc)
            self.add(self.path_title, pos=(1, 0), flag=rc)
            self.add(self.path_input, pos=(1, 1), flag=rc)
            self.add(self.path_count, pos=(1, 2), flag=rc)
            self.add(self.load_button, pos=(1, 3), flag=rc)
            self.add(self.vid_title, pos=(2, 0), flag=rc)
            self.add(self.video_name, pos=(2, 1), flag=rc)
            self.add(self.vid_input, pos=(2, 2), flag=rc)
            self.add(self.vid_modifier, pos=(2, 3), flag=wx.ALL|wx.EXPAND)
            self.vid_modifier.add(self.vid_inc, pos=(0, 0), flag=lc)
            self.vid_modifier.add(self.vid_dec, pos=(0, 2), flag=rc)
            self.vid_modifier.sizer.AddGrowableCol(1)
            self.add(self.fid_title, pos=(3, 0), flag=rc)
            self.add(self.frame_ord, pos=(3, 1), flag=rc)
            self.add(self.fid_input, pos=(3, 2), flag=rc)
            self.add(self.fid_modifier, pos=(3, 3), flag=wx.ALL|wx.EXPAND)
            self.fid_modifier.add(self.fid_inc, pos=(0, 0), flag=lc)
            self.fid_modifier.add(self.fid_dec, pos=(0, 2), flag=rc)
            self.fid_modifier.sizer.AddGrowableCol(1)

            self.load_button.button.Bind(wx.EVT_BUTTON, lambda e: load_videos(self.path_input.GetValue(), storage, config))
            self.vid_inc.Bind(wx.EVT_BUTTON, lambda e: video_idx_inc(storage, config))
            self.vid_dec.Bind(wx.EVT_BUTTON, lambda e: video_idx_dec(storage, config))
            self.vid_input.Bind(wx.EVT_KILL_FOCUS, lambda e: (video_idx_set(storage, config), e.Skip()))
            self.fid_inc.Bind(wx.EVT_BUTTON, lambda e: frame_idx_inc(storage, config))
            self.fid_dec.Bind(wx.EVT_BUTTON, lambda e: frame_idx_dec(storage, config))
            self.fid_input.Bind(wx.EVT_KILL_FOCUS, lambda e: (frame_idx_set(storage, config), e.Skip()))
            

    class LabelConfig(Panel):
        """
        标注配置面板
        """
        def __init__(self, parent):
            super().__init__(parent)
            ar = wx.ALIGN_RIGHT | wx.ALIGN_CENTER_VERTICAL
            self.sim_th_title = wx.StaticText(self, label="相似度阈值")
            self.sim_th_input = wx.TextCtrl(self, value="0.92", size=wx.Size(60, -1))
            self.sim_th_calculate = ButtonWithStatus(self, label="计算采样")
            self.sample_label_button = ButtonWithStatus(self, label="采样标注")
            self.sample_label_progress = wx.StaticText(self, label="0/0", size=wx.Size(60, -1), style=ar)
            self.tid_modifier = Panel(self, vgap=0, hgap=0)
            self.tid_inc = GenButton(self.tid_modifier, label="+", size=wx.Size(33, -1))
            self.tid_dec = GenButton(self.tid_modifier, label="-", size=wx.Size(33, -1))
            self.train_button = ButtonWithStatus(self, label="开始训练")
            # self.appending_train_button = GenButton(self, label="帧追加训练")
            self.auto_label = GenButton(self, label="自动标注")
            self.save_button = ButtonWithStatus(self, label="保存结果")

            lc = wx.ALIGN_CENTER_VERTICAL | wx.ALIGN_LEFT
            rc = wx.ALIGN_CENTER_VERTICAL | wx.ALIGN_RIGHT
            self.add(self.sim_th_title, pos=(0, 0), flag=rc)
            self.add(self.sim_th_input, pos=(0, 1), flag=rc)
            self.add(self.sim_th_calculate, pos=(0, 2), flag=rc)
            self.add(self.sample_label_progress, pos=(1, 1), flag=rc)
            self.add(self.sample_label_button, pos=(1, 2), flag=rc)
            self.add(self.tid_modifier, pos=(1, 4), flag=wx.ALL|wx.EXPAND)
            self.tid_modifier.add(self.tid_inc, pos=(0, 0), flag=lc)
            self.tid_modifier.add(self.tid_dec, pos=(0, 2), flag=rc)
            self.add(self.train_button, pos=(2, 2), flag=rc)
            # self.add(self.appending_train_button, pos=(2, 4), flag=rc)
            self.add(self.auto_label, pos=(3, 2), flag=rc)
            self.add(self.save_button, pos=(4, 2), flag=rc)
            self.sim_th_calculate.button.Bind(wx.EVT_BUTTON, lambda e: get_clip_features(storage, config))
            self.sample_label_button.button.Bind(wx.EVT_BUTTON, lambda e: start_sample_label(storage, config))
            self.tid_inc.Bind(wx.EVT_BUTTON, lambda e: train_idx_inc(storage, config))
            self.tid_dec.Bind(wx.EVT_BUTTON, lambda e: train_idx_dec(storage, config))
            self.train_button.button.Bind(wx.EVT_BUTTON, lambda e: sample_train(storage, config))
            # self.appending_train_button.Bind(wx.EVT_BUTTON, lambda e: append_train(storage, config))
            self.auto_label.Bind(wx.EVT_BUTTON, lambda e: start_auto_detect(storage, config))
            self.save_button.button.Bind(wx.EVT_BUTTON, lambda e: save_results(storage, config))
            top = self.GetTopLevelParent()
            top.Bind(wx.EVT_CHAR_HOOK, lambda e: global_key_handler(storage, config, e))

            self.timer = wx.Timer(self)
            self.Bind(wx.EVT_TIMER, lambda e: global_timer_handler(storage, config), self.timer)
            self.timer.Start(30)


    class FrameDisplay(Panel):
        """
        视频帧显示面板
        """
        def __init__(self, parent):
            super().__init__(parent)
            self.frame = wx.StaticBitmap(self, bitmap=wx.BitmapBundle(wx.Bitmap(config.frame_size[0], config.frame_size[1])))

            rc = wx.ALIGN_CENTER_VERTICAL | wx.ALIGN_RIGHT
            self.add(self.frame, pos=(0, 0), flag=rc, border=0)
    

    class Outputs(Panel):
        """
        输出信息面板
        // 包括全局进度条和文本输出
        """
        def __init__(self, parent):
            super().__init__(parent)
            self.progress = ProgressBar(self)
            self.output_field = RichTextCtrl(self, style=wx.TE_MULTILINE | wx.TE_READONLY)
            self.output_field.SetMinSize(wx.Size(-1, 80))
            self.sizer.AddGrowableCol(0)

            self.add(self.progress, pos=(0, 0), flag=wx.EXPAND|wx.ALL)
            self.add(self.output_field, pos=(1, 0), flag=wx.EXPAND|wx.ALL)

    
    class DetectFrame(MainFrame):
        """
        检测任务主程序窗口
        """
        def __init__(self, title, size):
            super().__init__(title, size)
            self.load_video_opts = LoadVideoOpts(self.frame_panel)
            self.label_config = LabelConfig(self.frame_panel)
            self.frame_display = FrameDisplay(self.frame_panel)
            self.outputs = Outputs(self.frame_panel)

            self.add(self.load_video_opts, pos=(0, 0))
            self.add(self.label_config, pos=(1, 0))
            self.add(self.frame_display, pos=(0, 1), span=(2, 1))
            self.add(self.outputs, pos=(2, 0), span=(1, 2))
            
            self.SetAutoLayout(True)
            self.set_sizer()
            storage.window = self
            append_output_text(storage, "=====主程序已启动=====", weight=wx.FONTWEIGHT_BOLD)
    

    DetectFrame("Label animals", size=(960, 540))

    app.MainLoop()
