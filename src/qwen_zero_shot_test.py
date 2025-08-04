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
from model import Qwen2VLForConditionalGeneration
from transformers import Qwen2VLProcessor
# from transformers import Qwen2VLProcessor, Qwen2VLForConditionalGeneration, AutoProcessor
from peft import PeftModel
from datasets import load_from_disk
from transformers import StoppingCriteria, StoppingCriteriaList
import re
Image.MAX_IMAGE_PIXELS = None

class StopOnBboxTag(StoppingCriteria):
    def __init__(self, tokenizer, stop_str="</bbox>"):
        # 把触发字符串编码成 token id 列表（不加任何 special tokens）
        self.tokenizer = tokenizer
        self.stop_ids = tokenizer.encode(stop_str, add_special_tokens=False)
        # 为了多批次也能用
        self.stop_len = len(self.stop_ids)

    def __call__(self, input_ids, scores, **kwargs):
        # input_ids: Tensor of shape (batch_size, seq_len)
        # 我们只关心最后 stop_len 个 id 是否和 stop_ids 一致
        if input_ids.shape[-1] < self.stop_len:
            return False
        # 对于每个序列都检查一下
        for seq in input_ids:
            if seq[-self.stop_len:].tolist() == self.stop_ids:
                return True
        return False

def extract_bbox(text, scale):
    # 用正则匹配 <bbox> 和 </bbox> 之间的 [x1,y1,x2,y2] 片段
    m = re.search(r'<bbox>\s*(\[[^\]]*\])\s*</bbox>', text)
    if not m:
        return None
    coords_str = m.group(1)               # e.g. "[12,34,56,78]"
    try:
        coords = json.loads(coords_str)   # 变成 [12, 34, 56, 78]
    except json.JSONDecodeError:
        # 如果你生成的数字里没有逗号分隔或者格式不标准，也可以手动 split
        nums = coords_str.strip('[]').split(',')
        coords = [float(x) for x in nums]
    return [x * scale for x in coords]

def extract_answer(text):
    m = re.search(r'<answer>\s*(.*?)\s*</answer>', text)
    if not m:
        return None
    answer = m.group(1)
    return answer

def extract_location(text):
    m = re.search(r'<location>\s*(.*?)\s*</location>', text)
    if not m:
        return None
    answer = m.group(1)
    return answer

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

def collate_fn(examples, processor, dtype, img_folder):

    texts = []
    images = []

    for example in examples:
        text = "<|image_pad|> \n" + example["question"]  # <image> how are you ?  # <|placeholder|> <|placeholder|> ...<|placeholder|>  how are you ?
        texts.append(text)
        images.append(resize_image(Image.open(img_folder + '/'+example["image_name"]).convert("RGB")))

    tokens = processor(text=texts, images=images, return_tensors="pt", padding="longest")
    # print('hello')
    return tokens.to(dtype), None  # 你可以用 id 也可以不返回

def prepare_dataloader(ds_path, collate_fn):
    mixed_dataset_val = load_from_disk(ds_path)
    val_dataloader = DataLoader(
        mixed_dataset_val,
        batch_size=1,
        collate_fn=collate_fn,
        shuffle=False,
        num_workers=1,
    )
    return val_dataloader

def chat(prompt, imgs, processor, accelerator, model, stop_criteria):
    inputs = processor(
        text=[prompt],
        images=imgs,
        return_tensors="pt",
        padding="longest"
    ).to(accelerator.device)

    gen_ids = model.generate(
        **inputs,
        max_new_tokens=1024,
        do_sample=False,
        num_beams=1,
        stopping_criteria=stop_criteria
    )
    
    gen_ids = gen_ids[:, inputs["input_ids"].shape[-1]:]
    output = processor.tokenizer.decode(
        gen_ids[0],
        skip_special_tokens=False
    ).strip()
    return output

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

    model_name: str = "/cephfs/shared/ruixun/project/lrx2/LRS_VQA/model"

    test_ds = load_from_disk("./LRS_VQA_RL/test")

    lrs_map = build_lrs_map("./LRS_VQA_merged.jsonl")

    model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name, device_map="auto", torch_dtype=torch.float16, trust_remote_code=True
        )
    # model = PeftModel.from_pretrained(model,
    #     "output_2/checkpoint-830",
    #     torch_dtype=torch.float16
    # )
    model.eval()

    processor = Qwen2VLProcessor.from_pretrained(
        model_name, trust_remote_code=True,
        max_pixels = 128*128*28*28
    )

    accelerator = Accelerator(mixed_precision="fp16", project_dir="checkpoints", log_with=[])
    model = accelerator.prepare_model(model)

    model.generation_config.temperature = None
    model.generation_config.top_p = None
    model.generation_config.top_k = None

    os.makedirs("results", exist_ok=True)
    out_path = "results/lora_infer_zero_shot.jsonl"
    fout = open(out_path, "w", encoding="utf-8")

    tokenizer = processor.tokenizer
    stop_criteria = StoppingCriteriaList([StopOnBboxTag(tokenizer, "</bbox>")])
    instruction = """
Task:
1. Global view – Give a one-sentence description of the entire scene.
2. Reasoning focus – Decide which part of the image you must attend to in order to answer the question. Wrap the chosen keyword (pick exactly one from bottom-left, bottom-right, bottom-center, top-left, top-right, top-center, center-left, center-right, center) in the tag <location>...</location>.
3. Answer box – Output the bounding box of that region as pixel coordinates in the form <bbox>[x1,y1,x2,y2]</bbox>. Use integers, no spaces.
4. Post-crop analysis - After cropping to the box in step 3, examine that patch and write a brief statement explaining the visual evidence that supports your answer.
5. Answer - your answer. In the tag <answer>...</answer>

Rules:
- Return exactly one <location> tag and one <bbox> tag; nothing else after them.
- If unsure, pick the most probable location and best-guess box—never say you are uncertain.
"""
    for ex in tqdm(test_ds):
        try:
            qid = ex["question_id"]
            scale = ex["scale"]
            bbox_ref = ex["bbox"]
            sample = lrs_map[qid]

            image_fp = "./image/"+sample["image"].split("/")[-1]
            cur_prompt = sample["text"]

            # stage 1: Question + Image (downsampled)
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": cur_prompt},
                    ],
                },
            ]

            # 加载图片，注意如果 image_path 是相对路径，需要拼接根目录
            image = Image.open(image_fp).convert("RGB")
            image = resize_image(image)
            
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=[text], images=[image], return_tensors="pt").to(model.device)

            generate_ids = model.generate(**inputs, max_new_tokens=128)
            outputs = processor.tokenizer.batch_decode(generate_ids, skip_special_tokens=True)[0].strip()

            record(fout, cur_prompt, sample, outputs)
        except:
            record(fout, cur_prompt, sample, "")
    fout.close()
    print("Done! 输出保存在", out_path)

if __name__ == "__main__":
    eval_model_lora()