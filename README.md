## 环境
1. 运行环境
    - OS==windows11（Linux没有实验过，自行配置）
    - python==3.10.0
    - cuda==12.8
2. pip install requirements.txt
3. 把下载的模型和视频数据文件放到相应的位置，见下面的目录树：
   ```
    AnimalsVideosProcess
    ├─.venv/...
    ├─.vscode/...
    ├─data
    │  ├─detect/...
    │  ├─metadata/...
    │  ├─truncated_videos/...
    │  │  ├─0000.mp4
    │  │  ├─0001.mp4
    │  │  └─...
    │  ├─videos/...
    │  └─videos_descriptions.csv
    ├─models
    │  ├─clip-vit-base-patch32
    │  └─clip_trt.pt2
    ├─src/...
    ├─temp/...
    ├─yolo11n.pt
    └─...
   ```

## 视频预处理
1. 将原始视频放置在data/videos目录下
2. 执行name_mapping.py，得到整理后的视频文件夹，与视频名称的映射存储在name_mapping.csv种
3. 执行get_metadata.py，得到视频元数据metadata.csv
3. 依次执行truncate.py、truncate2.py，裁剪视频，去除片头和片尾，truncate2.py用于裁剪特殊的5个片头
## 标注
1. 切换目录到本项目的根目录，然后python src/detect/interface.py
2. 输入元数据文件路径和视频文件夹路径（和默认的一样可以不修改）
3. 点击`加载视频`
4. 输入相似度阈值，通常设置0.85~0.90，一般标注20帧左右即可，然后点击`计算采样`
5. 点击`采样标注`，然后点击采样标注后的加减号来切换需要标注的视频帧
    - 按键`W`、`A`、`S`、`D`，可以更新标注框的大小
    - 按键`←`、`↑`、`↓`、`→`，可以更新标注框的位置
    - 按键`B`，可以切换标注框
    - 按键`C`，可以新建一个新的标注框
    - 按键`Delete`，可以删除当前标注框
    - 按键`T`，可以切换标注框的类别（类别从元数据文件自动加载）
6. 标注完成后，点击`开始训练`
7. 训练完成后，点击自动标注，然后长按`N`，自动标注，此时可以修改帧索引，来查看之前/之后的帧并进行修改
8. 全部标注完成后，点击`保存结果`
