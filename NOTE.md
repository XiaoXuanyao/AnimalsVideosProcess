## windows安装tensorrt

CUDA==12.8
python==3.10.0
torch==2.9.1+cu128

手动清空所有包残留，然后重新安装：

pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install transformers wxPython opencv-python matplotlib ultralytics

python -m pip install torch torch-tensorrt tensorrt==10.13.0.35 --extra-index-url https://download.pytorch.org/whl/cu128

pip install nvidia-modelopt[onnx]
pip install triton-windows

pip install tensorrt_llm --extra-index-url https://download.pytorch.org/whl/cu128


出现WARNING:torch_tensorrt [TensorRT Conversion Context]:Unable to determine GPU memory usage: In nvinfer1::getGpuMemStatsInBytes at common/extended/resources.cpp:1167，需要将tensorrt降版本至10.13.0.35
（CUDA==12.8）

安装TensorRT-LLM（未成功）：
解决git  Filename too long：git config --system core.longpaths true（管理员模式运行）
git clone -b release/0.21 https://github.com/NVIDIA/TensorRT-LLM.git
修改requirements.txt：
    nvidia-modelopt[torch]~=0.33.0
    删除nvidia-nccl-cu12