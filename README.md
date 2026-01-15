1. 将原始视频放置在data/videos目录下
2. 执行name_mapping.py，得到整理后的视频文件夹，与视频名称的映射存储在name_mapping.csv种
3. 执行get_metadata.py，得到视频元数据metadata.csv
3. 依次执行truncate.py、truncate2.py，裁剪视频，去除片头和片尾，truncate2.py用于裁剪特殊的5个片头
4. 