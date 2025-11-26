# conda create -n vlm-r1 python=3.11 
# conda activate vlm-r1

# Install the packages in open-r1-multimodal .
cd src/open-r1-multimodal # We edit the grpo.py and grpo_trainer.py in open-r1 repo.
pip install -e ".[dev]" -i https://pypi.tuna.tsinghua.edu.cn/simple

# Addtional modules
pip install wandb==0.18.3 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install tensorboardx -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install qwen_vl_utils torchvision -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install flash-attn --no-build-isolation -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install babel -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install python-Levenshtein -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install matplotlib -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install pycocotools -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install openai -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install httpx[socks] -i https://pypi.tuna.tsinghua.edu.cn/simple