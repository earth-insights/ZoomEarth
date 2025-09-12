# export CUDA_VISIBLE_DEVICES=7
import os
import json
from PIL import Image
import torch
from tqdm import tqdm
import shortuuid
from functools import partial
from torch.utils.data import DataLoader
from accelerate import Accelerator
# from model import Qwen2VLProcessor, Qwen2VLForConditionalGeneration
# from model import Qwen2VLForConditionalGeneration
# from transformers import Qwen2VLProcessor
from transformers import Qwen2VLProcessor, Qwen2_5_VLForConditionalGeneration, AutoProcessor
from peft import PeftModel
from datasets import load_from_disk
from transformers import StoppingCriteria, StoppingCriteriaList
import re
Image.MAX_IMAGE_PIXELS = None

def build_lrs_map(jsonl_path):
    """ 从 LRS_VQA_merged.jsonl 里读出所有样本，按 question_id 建映射 """
    mapping = {}
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            qid = item["question_id"]
            mapping[qid] = item
    return mapping

def cut_image(image, bbox, min_size=512):
    """
    最终输出统一为 min_size × min_size：
    - 若 bbox 边长 < min_size → 在原图中扩展，必要时平移避免越界；
    - 若 bbox 边长 >= min_size → 对 bbox 区域缩放并中心裁剪。
    """
    x1, y1, x2, y2 = map(int, bbox)
    width, height = x2 - x1, y2 - y1

    if width < min_size or height < min_size:
        # 中心点
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2

        # 初步计算边界
        new_x1 = center_x - min_size // 2
        new_y1 = center_y - min_size // 2
        new_x2 = new_x1 + min_size
        new_y2 = new_y1 + min_size

        # 平移使得裁剪框在图像内部
        if new_x1 < 0:
            new_x2 += -new_x1
            new_x1 = 0
        if new_y1 < 0:
            new_y2 += -new_y1
            new_y1 = 0
        if new_x2 > image.width:
            new_x1 -= new_x2 - image.width
            new_x2 = image.width
        if new_y2 > image.height:
            new_y1 -= new_y2 - image.height
            new_y2 = image.height

        # 最后确保框不越界
        new_x1 = max(0, new_x1)
        new_y1 = max(0, new_y1)
        new_x2 = min(image.width, new_x1 + min_size)
        new_y2 = min(image.height, new_y1 + min_size)

        return image.crop((int(new_x1), int(new_y1), int(new_x2), int(new_y2)))

    else:
        # 普通中心缩放逻辑
        cropped = image.crop((x1, y1, x2, y2))
        w, h = cropped.size
        scale = min_size / min(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cropped.resize((new_w, new_h), Image.BICUBIC)

        left = (new_w - min_size) // 2
        top = (new_h - min_size) // 2
        return resized.crop((left, top, left + min_size, top + min_size))

def resize_image(image, max_size=1024):
    w, h = image.size
    scale = max_size / max(w, h)
    if scale < 1:
        new_w = int(w * scale)
        new_h = int(h * scale)
        image = image.resize((new_w, new_h), Image.BICUBIC)
    return image

def record(fout, prompt, sample, output1):
    fout.write(json.dumps({
        "question_id":    sample["question_id"],
        "ground_truth":   sample["ground_truth"],
        "answer1":        output1,
        "prompt":         prompt,
        "category":       sample["category"],
        "stage1":         output1,
        "answer_id":      shortuuid.uuid(),
        "image":          sample["image"],
        "model_id":       "Qwen2-VL-7B-Instruct"
    }, ensure_ascii=False) + "\n")

def eval_model_lora():

    model_name: str = "Qwen/Qwen2.5-VL-3B-Instruct"

    test_ds = load_from_disk("./LRS_VQA_RL/test")

    lrs_map = build_lrs_map("/home/zhangjunjie/CaoXiongyong_Student/fubowen/LRS_VQA/LRS_VQA/LRS_VQA_merged.jsonl")

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name, device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True
    )
    model.eval()

    processor = AutoProcessor.from_pretrained(
        model_name, trust_remote_code=True,
    )

    model.generation_config.temperature = None
    model.generation_config.top_p = None
    model.generation_config.top_k = None

    os.makedirs("results", exist_ok=True)
    out_path = "results/lora_infer_zero_shot_cot.jsonl"
    fout = open(out_path, "a", encoding="utf-8")

    # cnt = 0
    for ex in tqdm(test_ds):
        # if cnt <= 179:
            # cnt+=1
            # continue
        # try:
        qid = ex["question_id"]
        sample = lrs_map[qid]

        image_fp = "./LRS_VQA/image/"+sample["image"].split("/")[-1]

        instruction = '''
## instruction
You are an intelligent remote sensing analyst. Your task is to answer visual questions about satellite images. 
For each question, follow the reasoning chain and provide the object of interest and its bounding box before giving the final answer.
Use english only.

Follow this structure exactly:

<think>
- Stage 1 Reasoning: Explain how you locate the object based on the question and visual features.
</think>
<locate>
- Object: Clearly describe the object you need to find.
- Bounding Box: Provide a JSON array with the 2D bounding box in pixel coordinates, format:
```json
[
  {"bbox_2d": [x_min, y_min, x_max, y_max], "label": "<object description>"}
]
```
</locate>
<think>
Stage 2 Reasoning: Describe how to answer the question based on the object you located.
<\think>
<answer> your answer (a single word or phrase.) </answer>

## Now apply to a new question

'''

        cur_prompt = sample["text"]

        # stage 1: Question + Image (downsampled)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": instruction + cur_prompt},
                ],
            },
        ]

        # 加载图片，注意如果 image_path 是相对路径，需要拼接根目录
        image = Image.open(image_fp).convert("RGB")
        image = resize_image(image)
        
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[image], return_tensors="pt").to(model.device)

        generate_ids = model.generate(**inputs, max_new_tokens=1024)
        outputs = processor.decode(generate_ids[0], skip_special_tokens=True).strip()

        record(fout, cur_prompt, sample, outputs)
        # except:
            # record(fout, cur_prompt, sample, "")
    fout.close()
    print("Done! 输出保存在", out_path)

if __name__ == "__main__":
    eval_model_lora()