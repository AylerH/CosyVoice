
# 安装
## 通用步骤
```
git clone --recursive https://github.com/aylerh/CosyVoice.git
# If you failed to clone the submodule due to network failures, please run the following command until success
cd CosyVoice
git submodule update --init --recursive
```
## windows-安装虚拟环境cosyvoice:
参考教程：
```
https://doupoa.site/archives/581
```
```
conda create -n cosyvoice -y python=3.10
conda activate cosyvoice
conda install -y -c conda-forge pynini==2.1.5

pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host=mirrors.aliyun.com
# windows


# If you encounter sox compatibility issues
# ubuntu
sudo apt-get install sox libsox-dev
# centos
sudo yum install sox sox-devel

```

# 运行
## 下载cosyvocie2-0.5B
```
python cosyvoice_download.py
```
## tts运行
```
python cosyvoice_tts.py
```
## web运行
```
python webui.py --port 50000 --model_dir pretrained_models/CosyVoice2-0.5B
```