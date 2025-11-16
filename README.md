<div align="center">
  <h1 style="display: flex; align-items: center; justify-content: center; gap: 10px;">
    ZoomEarth: Active Perception for Ultra-High-Resolution Geospatial Vision-Language Tasks
  </h1>
  <p>
  •
  <a href="https://earth-insights.github.io/ZoomEarth/">[Project]</a>
  •
  <a href="https://earth-insights.github.io/ZoomEarth/">[arXiv]</a>
  •
  </p>
</div>

## 🔥🔥🔥 ZoomEarth

We released ZoomEarth🌍, a vision language model that is designed to solve visual reasoning and question answering tasks on **ultra-high-resolution remote sensing imagery** with active perception. Moreover, ZoomEarth can seamlessly integrate with downstream models for tasks such as cloud removal, denoising, segmentation, and image editing through simple tool interfaces, demonstrating strong extensibility.  

## 🎊 News and Updates

- `2025.11.15` 🎉🎉🎉 ZoomEarth-3B is publicly available on [huggingface🤗](https://huggingface.co/HappyBug/ZoomEarth-3B)!
- `2025.11.15` 🎉🎉🎉 LRS-GRO is publicly available on [huggingface🤗](https://huggingface.co/datasets/HappyBug/LRS-GRO)!
- `2025.11.15` 🎉🎉🎉 ***ZoomEarth: Active Perception for Ultra-High-Resolution Geospatial Vision-Language Tasks*** is now avilable on [arXiv]()!

## 🧠 Model

Our model, **ZoomEarth**, is built upon Qwen2.5-VL-3B, a powerful VLM that  
It supports fine-grained reasoning, spatial context interpretation, and multi-level object understanding.

- Model weights: [ZoomEarth-3B 🤗](https://huggingface.co/HappyBug/ZoomEarth-3B)
- Training scripts: [./src/train/](./src/train/)  
- Evaluate scripts: [./src/eval/](./src/eval/)

## 🛰️ Dataset

**LRS-GRO** contains high-resolution satellite images annotated with:
- Multi-level question types (global, regional, object)
- Bounding boxes and spatial relations
- Reasoning-based and factual QAs  

| Split | #Images | #Questions | Avg. Resolution |
|:------|---------:|------------:|----------------:|
| SFT | 88 | 1011 | 5000 |
| RL  | 228 | 2500 | 5000 |
| Test| 908 | 9734 | 5000 |

**Download:**  
[LRS-GRO 🤗](https://huggingface.co/datasets/HappyBug/LRS-GRO)

## ⚙️ Installation
Step 1. Create a conda environment and activate it.
```
conda create -n zoom-earth python=3.10 -y
conda activate zoom-earth
```
Step 2. Install PyTorch (We use PyTorch 2.4.1 / CUDA 12.1)
```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```
Step 3. Install other depencencies 
```
pip install -r requirements.txt
```
Step 4. Configure NLTK local corpora (for WordNet)
```
# (1) Download WordNet data to a local directory (optional if already exists)
python -m nltk.downloader wordnet -d ./nltk_data

# (2) In your code, add the following before importing WordNet
import nltk

local_corpora = "./nltk_data"
nltk.data.path.insert(0, local_corpora)

from nltk.corpus import wordnet as wn
```
and then replace `local_corpora` with actual path in [`src/eval/eval.py`](src/eval/eval.py), [`src/train/RL/src/open-r1-multimodal/src/open_r1/custom/customized_funcs.py`](src/train/RL/src/open-r1-multimodal/src/open_r1/custom/customized_funcs.py)
## 📋 Quick start
```python
python src/demo.py
```


## 🚂 Train
To train ZoomEarth, first run `bash ./run_scripts/train_sft.sh` to start SFT training phase.

After that, run `bash ./run_scripts/train_rl.sh` to start RL training phase.

## 📋 Test
To evaluate model on LRS-GRO, first run `bash ./run_scripts/infer.sh` to generate inference file.

After that, run `bash ./run_scripts/eval.sh` to get detailed evaluation result.
## 📬 Contact

If you have questions or would like to collaborate, please contact us at:  
📧 [`liuruixun6343@gmail.com`](liuruixun6343@gmail.com)

📧 [`HappyBug@stu.xjtu.edu.cn`](HappyBug@stu.xjtu.edu.cn)

<p align="center">
  <sub>© 2025 ZoomEarth Project. Released under the Apache 2.0 License.</sub>
</p>
