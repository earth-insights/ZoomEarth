## Usage: run `accelerate config` in shell to config accelerate first, 
## then `accelerate launch --num_processes=8 src/qwen_rl.py` to train

import copy
import json
import re
from typing import List
import os

import torch
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

# from model import Qwen2VLForConditionalGeneration
from transformers import Qwen2VLProcessor
from model import Qwen2VLForConditionalGeneration
# from transformers import Qwen2VLProcessor, Qwen2VLForConditionalGeneration
# from model import Qwen2VLProcessor, Qwen2VLForConditionalGeneration

from datasets import load_from_disk
from PIL import Image

from peft import PeftModel
Image.MAX_IMAGE_PIXELS = None
import random
from accelerate import Accelerator
from tqdm import tqdm
from datetime import datetime
from peft import LoraConfig


# =============== NLTK ===============
import nltk
local_corpora = os.path.join(os.getcwd())
nltk.data.path.insert(0, local_corpora)
from nltk.corpus import wordnet as wn
_WORDNET_OK = True

# ======================= 全局配置 =======================
LR = 2e-5
# MAX_STEPS = 2000
MAX_ITER = 5
NUM_GENERATIONS = 1        # G：每个 prompt 采样条数
INNER_GRPO_STEPS = 2       # 复用同一批样本做 K 次反向
BETA_KL = 0.01             # KL 系数
REF_SYNC_INTERVAL = 500    # 参考策略同步周期（步）

lora_config = LoraConfig(
    r=8,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"],
    task_type="CAUSAL_LM",
)

# ======================= 懒加载 / 缓存 =======================
_TAG_RE_CACHE = {}

# ======================= Log =======================
PRINT_EVERY = 50
LOG_EVERY   = 50
SAVE_EVERY = 400

# ======================= 数据集 & 组装器 =======================
def collate_fn(examples):
    return examples[0]

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

# ======================= 图片操作: 下采样, 裁剪 =======================

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

def make_vision_kwargs(imgs, processor, device):
    tmp = processor(text=["<dummy>"], images=imgs, return_tensors="pt", padding="longest")
    vk = {k: v.to(device) for k, v in tmp.items() if k not in ("input_ids","attention_mask")}
    return vk

# ======================= GT / 奖励工具 =======================
def _lemmatize(word:str) -> str:
    try:
        from nltk.stem import WordNetLemmatizer
        return WordNetLemmatizer().lemmatize(word)
    except Exception:
        return word

def are_synonyms(word1:str, word2:str) -> bool:
    if not _WORDNET_OK:
        return False
    w1, w2 = _lemmatize(word1.lower()), _lemmatize(word2.lower())
    try:
        synsets1 = wn.synsets(w1)
        synsets2 = wn.synsets(w2)
    except Exception:
        return False
    best = 0.0
    for s1 in synsets1:
        for s2 in synsets2:
            sim = s1.path_similarity(s2)
            if sim is not None and sim > best:
                best = sim
    return best >= 0.8

def is_correct(answer:str, gt:str) -> bool:
    if answer is None:
        answer = ""
    answer = answer.strip().lower().rstrip('.')
    gt = (gt or "").strip().lower().rstrip('.')
    if not gt:
        return False
    return (answer == gt) or are_synonyms(gt, answer)

# ============== 结构化标签解析 / BBox & IoU ==============

def extract_tag(text: str, tag: str, default=None):
    if tag not in _TAG_RE_CACHE:
        safe = re.escape(tag)
        _TAG_RE_CACHE[tag] = re.compile(rf'<{safe}\s*>\s*(.*?)\s*</{safe}\s*>', re.S)
    m = _TAG_RE_CACHE[tag].search(text or "")
    return m.group(1).strip() if m else default

def parse_bbox_text(s):
    if s is None:
        return None
    # 优先 JSON
    try:
        arr = json.loads(s)
        if isinstance(arr, list) and len(arr) == 4:
            return [float(v) for v in arr]
    except Exception:
        pass
    # 兜底：抓 4 个数
    nums = re.findall(r'-?\d+\.?\d*', s)
    if len(nums) >= 4:
        return [float(x) for x in nums[:4]]
    return None

def _fix_order(box):
    x1, y1, x2, y2 = box
    return [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]

def get_crop_area(bbox, min_size=512):
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

        # 最后确保框不越界
        new_x1 = max(0, new_x1)
        new_y1 = max(0, new_y1)

        return [int(new_x1), int(new_y1), int(new_x2), int(new_y2)]
    else:
        return bbox

def iou(box_a, box_b):
    box_a = get_crop_area(box_a)
    box_b = get_crop_area(box_b)
    x1a, y1a, x2a, y2a = _fix_order(box_a)
    x1b, y1b, x2b, y2b = _fix_order(box_b)

    inter_x1, inter_y1 = max(x1a, x1b), max(y1a, y1b)
    inter_x2, inter_y2 = min(x2a, x2b), min(y2a, y2b)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, (x2a - x1a)) * max(0.0, (y2a - y1a))
    area_b = max(0.0, (x2b - x1b)) * max(0.0, (y2b - y1b))

    union = area_a + area_b - inter_area
    return 0.0 if union == 0 else inter_area / union

# ========================== 奖励函数 ==========================

def score_one(completion1:str, completion2:str, ground_truth:dict) -> float:
    if not isinstance(ground_truth, dict):
        return 0.0

    bbox = parse_bbox_text(extract_tag(completion1, "bbox"))
    answer = extract_tag(completion2, "answer")
    location = extract_tag(completion1, "location")

    bbox_ref = ground_truth.get("bbox")
    location_ref = ground_truth.get("area")
    answer_ref = ground_truth.get("ground_truth")
    cut = bool(ground_truth.get("cut", False))

    if not cut:
        return 1.0 if is_correct(answer, answer_ref) else 0.0

    pattern_reward = 1.0 if (bbox is not None and answer and location) else 0.0

    # bbox reward
    bbox_reward = 0.0
    if isinstance(bbox_ref, (list, tuple)) and len(bbox_ref) == 4 and isinstance(bbox, (list, tuple)) and len(bbox) == 4:
        cx, cy = (bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0
        rx, ry = (bbox_ref[0] + bbox_ref[2]) / 2.0, (bbox_ref[1] + bbox_ref[3]) / 2.0
        distance = ((rx - cx) ** 2 + (ry - cy) ** 2) ** 0.5 + 1e-6
        bbox_iou = iou(bbox, bbox_ref)
        bbox_reward = float(bbox_iou + 1.0 / distance)

    # answer reward
    answer_reward = 1.0 if is_correct(answer, answer_ref) else 0.0

    # location reward
    loc_reward = 0.0
    if isinstance(location_ref, str) and isinstance(location, str):
        if location_ref == "center-":
            loc_reward += 2.0 if location == "center-" else 0.0
        else:
            loc_parts = location.split("-")
            ref_parts = location_ref.split("-")
            if len(loc_parts) >= 1 and len(ref_parts) >= 1 and loc_parts[0] == ref_parts[0]:
                loc_reward += 1.0
            if len(loc_parts) >= 2 and len(ref_parts) >= 2 and loc_parts[1] == ref_parts[1]:
                loc_reward += 1.0

    return float(pattern_reward + bbox_reward + answer_reward + loc_reward)

def advantage_fun_single(gt, completions1:List[str], completions2:List[str], normalize: str = "zscore") -> torch.FloatTensor:
    rewards = [score_one(completions1[i], completions1[i], gt) for i in range(len(completions1))]
    t = torch.tensor(rewards, dtype=torch.float32)
    if normalize == "mean":
        t = t - t.mean()
    elif normalize == "zscore":
        t = (t - t.mean()) / (t.std(unbiased=False) + 1e-6)
    elif normalize in (None, "none"):
        pass
    else:
        raise ValueError(f"Unknown normalize={normalize}")
    return t

def advantage_fun(gt, completions1:List[str], completions2:List[str], normalize:str="zscore") -> torch.FloatTensor:
    chunks = []
    chunks.append(advantage_fun_single(gt, completions1, completions2, normalize=normalize))
    return torch.cat(chunks, dim=0)

# ========================== 生成 & 封装 ==========================

def chat(prompt, imgs, processor, accelerator, model, temperature):
    inputs = processor(
        text=[prompt],
        images=imgs,
        return_tensors="pt",
        padding="longest"
    ).to(accelerator.device)

    gen_ids = accelerator.unwrap_model(model).generate(
        **inputs,
        max_new_tokens=1024,
        do_sample=True,
        num_beams=1,
        temperature=temperature,
        top_p=0.9
    )

    prompt_len = inputs["input_ids"].shape[-1]
    full_ids = gen_ids[0].detach().cpu()
    gen_ids = gen_ids[:, prompt_len:]
    output = processor.tokenizer.decode(
        gen_ids[0],
        skip_special_tokens=False
    ).strip()

    return full_ids, prompt_len, output

@torch.no_grad()
def generate_completion_vl_single(gt, processor, accelerator, model, temperature):
    model.eval()
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
    image_fp = "./image/"+gt["image_name"].split("/")[-1]
    scale = gt["scale"]
    cur_prompt = gt["question"].split("Answer the question using a single word or phrase.")[0]

    # stage 1: Question + Image (downsampled)
    img = Image.open(image_fp).convert("RGB")
    img = resize_image(img)

    prompt =  "<|image_pad|>\n" + cur_prompt + instruction
    ids1, plen1, output1 = chat(prompt, [img], processor, accelerator, model, temperature)

    # stage 2: Question + Image (downsampled) + previous reasoning + Image (cropped)
    prompt =  prompt + output1.split("<answer>")[0] + "<|image_pad|>"
    bbox = parse_bbox_text(extract_tag(output1, "bbox"))
    if bbox is None:
        W, H = img.size
        bbox = [0, 0, W, H]
    bbox = [x * scale for x in bbox]
    image_bbox = Image.open(image_fp).convert("RGB")
    image_bbox = cut_image(image_bbox, bbox)
    image_bbox = resize_image(image_bbox)

    ids2, plen2, out2 = chat(prompt, [img, image_bbox], processor, accelerator, model, temperature)
    vision_kwargs = make_vision_kwargs([img, image_bbox], processor, accelerator.device)
    return ids2, plen2, output1, out2, vision_kwargs

@torch.no_grad()
def generate_completion_vl(gt, processor, accelerator, model, G):
    temp_center = 0.01  # 中心温度
    temp_range = 0.005  # 波动幅度

    ids_list = []
    plen_list = []
    out2_list = []
    out1_list = []
    vision_kwargs = None
    for i in range(G):
        temp = temp_center + random.uniform(-temp_range, temp_range)
        temp = max(0.0, temp)  # 确保非负
        ids, plen, out1, out2, vk = generate_completion_vl_single(gt, processor, accelerator, model, temperature=temp)
        ids_list.append(ids)
        plen_list.append(plen)
        out1_list.append(out1)
        out2_list.append(out2)
        if vision_kwargs is None:
            vision_kwargs = vk
    return ids_list, plen_list, out1_list, out2_list, vision_kwargs


# ============== Log p 序列概率（批处理，支持反向） ==============

def _pack_batch(batch_ids:List[torch.Tensor], pad_id:int, device:str):
    lens = [int(x.numel()) for x in batch_ids]
    B = len(batch_ids)
    T = max(lens)
    x = torch.full((B, T), pad_id, dtype=torch.long, device=device)
    mask = torch.zeros((B, T), dtype=torch.long, device=device)
    for i, ids in enumerate(batch_ids):
        L = lens[i]
        x[i, :L] = ids.to(device)
        mask[i, :L] = 1
    return x, mask, lens

# def batch_seq_logprobs_vl(model, batch_ids:List[torch.Tensor], prompt_lens:List[int], pad_id:int, device:str, no_grad:bool, vision_kwargs:dict) -> torch.Tensor:
#     ctx = torch.no_grad() if no_grad else torch.enable_grad()
#     with ctx:
#         x, mask, lens = _pack_batch(batch_ids, pad_id, device)

#         num_text = len(batch_ids)       # 文本 batch 大小
#         num_img = vision_kwargs["pixel_values"].shape[0]  # 原图像数量

#         if num_text > num_img:
#             repeat_factor = num_text // num_img
#             # repeat_interleave 更清晰：第0维复制 repeat_factor 次
            
#             vision_kwargs = {
#                 k: v.repeat_interleave(repeat_factor, dim=0)
#                 for k, v in vision_kwargs.items()
#             }
#             print("num_text:", num_text, "num_img:", num_img)
#             print("repeat_factor:", repeat_factor)
#             for k, v in vision_kwargs.items():
#                 print(f"{k}.shape after repeat: {v.shape}")


#         out = model(input_ids=x, attention_mask=mask, **vision_kwargs)
#         logits = out.logits[:, :-1, :]
#         labels = x[:, 1:]
#         valid = mask[:, 1:].bool()

#         logp = torch.log_softmax(logits, dim=-1)
#         tok_lp = logp.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
        
#         Bsz, Tm1 = labels.shape
#         pos = torch.arange(Tm1, device=device).unsqueeze(0).expand(Bsz, -1)
#         pl = torch.as_tensor(prompt_lens, device=device).unsqueeze(1)
#         gen_mask = pos >= (pl - 1)
#         final_mask = valid & gen_mask & (labels != pad_id)
#         seq_logp = (tok_lp * final_mask).sum(dim=1)
#     return seq_logp

def batch_seq_logprobs_vl(model, batch_ids, prompt_lens, pad_id, device, no_grad, vision_kwargs):
    num_text = len(batch_ids)  # 比如 G=4，4个sample
    # 直接把视觉特征沿 batch 维度 repeat num_text 次
    vision_kwargs = {
        k: v.repeat(num_text, *([1] * (v.ndim - 1))) for k, v in vision_kwargs.items()
    }
    # print(f"[DEBUG] vision_kwargs expanded to:")
    for k, v in vision_kwargs.items():
        print(f"  {k}: {v.shape}")

    ctx = torch.no_grad() if no_grad else torch.enable_grad()
    with ctx:
        x, mask, lens = _pack_batch(batch_ids, pad_id, device)
        # print(f"[DEBUG] x.shape={x.shape}, mask.shape={mask.shape}")

        out = model(input_ids=x, attention_mask=mask, **vision_kwargs)
        logits = out.logits[:, :-1, :]
        labels = x[:, 1:]
        valid = mask[:, 1:].bool()

        logp = torch.log_softmax(logits, dim=-1)
        tok_lp = logp.gather(-1, labels.unsqueeze(-1)).squeeze(-1)

        Bsz, Tm1 = labels.shape
        pos = torch.arange(Tm1, device=device).unsqueeze(0).expand(Bsz, -1)
        pl = torch.as_tensor(prompt_lens, device=device).unsqueeze(1)
        gen_mask = pos >= (pl - 1)
        final_mask = valid & gen_mask & (labels != pad_id)
        seq_logp = (tok_lp * final_mask).sum(dim=1)
    return seq_logp

# def batch_seq_logprobs_vl(model, batch_ids, prompt_lens, pad_id, device, no_grad, vision_kwargs):
#     # 把视觉特征搬到计算设备
#     vision_kwargs = {
#         k: (v.to(device) if isinstance(v, torch.Tensor) else v)
#         for k, v in vision_kwargs.items()
#     }

#     # repeat 视觉特征
#     num_text = len(batch_ids)  # 比如 G=4
#     vision_kwargs = {
#         k: v.repeat(num_text, *([1] * (v.ndim - 1)))
#         for k, v in vision_kwargs.items()
#     }
#     print(f"[DEBUG] vision_kwargs expanded to:")
#     for k, v in vision_kwargs.items():
#         print(f"  {k}: {v.shape} @ {v.device}")

#     # 把文本 token 搬到计算设备
#     batch_ids = [t.to(device) for t in batch_ids]

#     ctx = torch.no_grad() if no_grad else torch.enable_grad()
#     with ctx:
#         x, mask, lens = _pack_batch(batch_ids, pad_id, device)
#         print(f"[DEBUG] x.shape={x.shape}, mask.shape={mask.shape}")

#         out = model(input_ids=x, attention_mask=mask, **vision_kwargs)
#         logits = out.logits[:, :-1, :]
#         labels = x[:, 1:]
#         valid = mask[:, 1:].bool()

#         logp = torch.log_softmax(logits, dim=-1)
#         tok_lp = logp.gather(-1, labels.unsqueeze(-1)).squeeze(-1)

#         Bsz, Tm1 = labels.shape
#         pos = torch.arange(Tm1, device=device).unsqueeze(0).expand(Bsz, -1)
#         pl = torch.as_tensor(prompt_lens, device=device).unsqueeze(1)
#         gen_mask = pos >= (pl - 1)
#         final_mask = valid & gen_mask & (labels != pad_id)
#         seq_logp = (tok_lp * final_mask).sum(dim=1)

#     # 计算完后搬回 CPU，节省显存
#     return seq_logp.cpu()

def save_lora_adapter(accelerator, model, path):
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.save_pretrained(
        path,
        is_main_process=accelerator.is_main_process,
        save_function=accelerator.save,
    )

def save_checkpoint(accelerator, model, epoch, step, loss, checkpoint_dir="./output_rl"):
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    save_lora_adapter(accelerator, model, checkpoint_dir)
    
    training_info = {
        "epoch": epoch,
        "step": step,
        "loss": loss,
        "latest_checkpoint": checkpoint_dir,
    }
    with open(f"{checkpoint_dir}/training_info.json", "w") as f:
        json.dump(training_info, f)

def set_lora_trainable(model, lora_config):
    """
    冻结所有参数，只解冻包含 lora_config.target_modules 中任意模块名的参数
    """
    target_modules = [m.lower() for m in lora_config.target_modules]

    for param in model.parameters():
        param.requires_grad = False

    for name, param in model.named_parameters():
        name_low = name.lower()
        if any(tm in name_low for tm in target_modules):
            param.requires_grad = True

def load_lora_weights(base_model, lora_state_dict, device, lora_config):
    """
    只加载 LoRA 权重，不影响基础权重，设置LoRA层为可训练
    """
    base_model = base_model.to("cpu")  # 先切到cpu方便加载
    base_model.load_state_dict(lora_state_dict, strict=False)
    base_model = base_model.to(device)
    set_lora_trainable(base_model, lora_config)
    return base_model

def extract_lora_state_dict(model, lora_config):
    """
    提取包含 target_modules 中模块名的权重，作为 LoRA 权重
    """
    target_modules = [m.lower() for m in lora_config.target_modules]
    lora_state_dict = {}

    for k, v in model.named_parameters():
        k_low = k.lower()
        if any(tm in k_low for tm in target_modules):
            lora_state_dict[k] = v.detach().cpu().clone()

    return lora_state_dict


# =============================== 主训练 ===============================

def main():
    accelerator = Accelerator(mixed_precision="bf16", project_dir="checkpoints_rl", gradient_accumulation_steps=INNER_GRPO_STEPS, log_with=["tensorboard"])

    accelerator.init_trackers(
        "grpo",
        init_kwargs={
            "wandb": {"name": f"{datetime.now().strftime('%Y-%m-%d %H-%M-%S')}"},
            "tensorboard": {}
        },
        config={
            "beta_kl": BETA_KL,
            "G": NUM_GENERATIONS,
            "K": INNER_GRPO_STEPS,
            "lr": LR,
            "max_iteration": MAX_ITER,
        },
    )

    running_loss = torch.zeros((), device=accelerator.device)
    running_count = 0

    # ==== 1) 模型 & Processor（θ 用 VL + LoRA；Ref 深拷贝 θ 并冻结） ====
    model_name: str = "/cephfs/shared/ruixun/project/lrx2/LRS_VQA/model"
    # model_name: str = "/home/zhangjunjie/.cache/huggingface/hub/models--Qwen--Qwen2-VL-7B-Instruct/snapshots/eed13092ef92e448dd6875b2a00151bd3f7db0ac"
    base_model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_name, torch_dtype=torch.float16, trust_remote_code=True
    )
    # 如果你要继续在 LoRA 上做 GRPO（推荐），加载 LoRA 适配器：
    base_model = PeftModel.from_pretrained(
        base_model, "./output_2/checkpoint-830", torch_dtype=torch.float16
    )

    # 参考策略：拷贝 θ 的当前权重，并冻结
    theta_adapter = extract_lora_state_dict(base_model, lora_config)

    ref_adapter = extract_lora_state_dict(base_model, lora_config)

    old_adapter = extract_lora_state_dict(base_model, lora_config)

    processor = Qwen2VLProcessor.from_pretrained(
        model_name, trust_remote_code=True, max_pixels=128*128*28*28
    )

    for name, p in base_model.named_parameters():
        if "lora" in name:
            p.requires_grad = True

    # 只优化可训练参数（LoRA 情况下就是适配器权重）
    optim = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, base_model.parameters()),
        lr=LR, weight_decay=0.01
    )

    # ==== 2) 数据 ====
    dl = prepare_dataloader("./LRS_VQA_RL/rl", collate_fn)

    base_model, optim, dl = accelerator.prepare(base_model, optim, dl)
    base_model.train()

    MAX_STEPS = MAX_ITER * len(dl) * INNER_GRPO_STEPS

    if accelerator.is_main_process:
        pbar = tqdm(total=MAX_STEPS, desc="GRPO", dynamic_ncols=True)
    
    sched = get_linear_schedule_with_warmup(optim, int(0.05 * MAX_STEPS), MAX_STEPS)

    # ==== 3) 训练循环 ====
    global_step = 0
    for iter in range(MAX_ITER):
        for example in dl:
            # ---- 采样：每个样本重复 G 次（组内比较）----
            with torch.no_grad():
                theta = load_lora_weights(base_model, theta_adapter, accelerator.device, lora_config)
                batch_ids, prompt_lens, completions1, completions2, vision_kwargs = generate_completion_vl(
                    example, processor, accelerator, theta, NUM_GENERATIONS
                )
            
            rewards = torch.tensor([score_one(completions1[i], completions2[i], example) for i in range(len(completions1))],
                                    device=accelerator.device, dtype=torch.float32)
            reward_mean = rewards.mean()
            reward_std  = rewards.std(unbiased=False)
            adv = (rewards - reward_mean) / (reward_std + 1e-8)
            
            reward_mean = reward_mean.item()
            reward_std = reward_std.item()

            tok = processor.tokenizer
            pad_id_eff = tok.pad_token_id
            if pad_id_eff is None:
                tok.pad_token = tok.eos_token
                pad_id_eff = tok.pad_token_id

            # ---- 缓存 old 对数似然（采样时刻的 θ，与 Ref）----
            with torch.no_grad():
                old = load_lora_weights(base_model, old_adapter, accelerator.device, lora_config)
                logp_old = batch_seq_logprobs_vl(
                    old, batch_ids, prompt_lens, pad_id_eff, accelerator.device, no_grad=True, vision_kwargs=vision_kwargs
                )
                torch.cuda.empty_cache()
                # print("\n\nstage1 done!\n\n")

                ref = load_lora_weights(base_model, ref_adapter, accelerator.device, lora_config)
                logp_ref = batch_seq_logprobs_vl(
                    ref, batch_ids, prompt_lens, pad_id_eff, accelerator.device, no_grad=True, vision_kwargs=vision_kwargs
                )
                torch.cuda.empty_cache()
                # print("\n\nstage2 done!\n\n")

            # ---- 内循环 K 次（重复反向，提高样本利用率）----
            theta = load_lora_weights(base_model, theta_adapter, accelerator.device, lora_config)
            for i in range(INNER_GRPO_STEPS):
                with accelerator.accumulate(theta):
                    theta.train()
                    
                    logp_theta = batch_seq_logprobs_vl(
                        theta, batch_ids, prompt_lens, pad_id_eff, accelerator.device, no_grad=False, vision_kwargs=vision_kwargs
                    )

                    # 比例项（REINFORCE with per-seq log prob）
                    ratio = torch.exp(logp_theta - logp_old)
                    policy_loss = -(ratio * adv).mean()

                    # GRPO 的简化 KL 惩罚（θ 对 ref）
                    # approx_kl = (logp_theta - logp_ref).mean()
                    # kl_loss = BETA_KL * approx_kl
                    
                    approx_kl = (logp_theta - logp_ref).pow(2).mean()
                    kl_loss = BETA_KL * approx_kl

                    loss = policy_loss + kl_loss
                    accelerator.backward(loss)

                    if accelerator.sync_gradients:
                        grad_norm = accelerator.clip_grad_norm_(theta.parameters(), 1.0)  # ← 用 accelerator 的裁剪以兼容分布式/张量并行
                        optim.step()
                        sched.step()
                        optim.zero_grad(set_to_none=True)
                    # print(f"\n\inner loop {i} done!\n\n")

                if accelerator.sync_gradients:
                    # 1) 计步 & 进度条
                    global_step += 1
                    if accelerator.is_main_process:
                        pbar.update(1)

                    # 2) 累加 running loss（用于 log 的滑动均值）
                    running_loss += loss.detach()
                    running_count += 1

                    # 3) 打印（轻量）
                    if global_step % PRINT_EVERY == 0:
                        accelerator.print(
                            f"[step {global_step}] "
                            f"loss={loss.item():.4f}  policy={policy_loss.item():.4f}  kl={approx_kl.item():.4f}  "
                            f"R_mean={reward_mean:.3f}  R_std={reward_std:.3f}  "
                            f"lr={sched.get_last_lr()[0]:.2e}"
                        )

                    # 4) 记录到 tracker（重一点）
                    if global_step % LOG_EVERY == 0:
                        try:
                            gnorm = grad_norm.detach().item() if isinstance(grad_norm, torch.Tensor) else float(grad_norm)
                        except NameError:
                            gnorm = 0.0

                        mean_loss_world = (accelerator.gather(running_loss).sum().item() /
                                        max(1, accelerator.num_processes) / max(1, running_count))

                        accelerator.log(
                            {
                                "train/loss": mean_loss_world,
                                "train/policy_loss": policy_loss.detach().float().mean().item(),
                                "train/kl": approx_kl.detach().float().item(),
                                "train/approx_kl": approx_kl.item(),
                                "train/learning_rate": sched.get_last_lr()[0],
                                "train/grad_norm": gnorm,
                                "reward/mean": reward_mean,
                                "reward/std": reward_std,
                                "train/global_step": global_step,
                            },
                            step=global_step,
                        )
                        running_loss.zero_()
                        running_count = 0

                    # 5) 可选：周期性保存
                    if global_step % SAVE_EVERY == 0 and accelerator.is_main_process:
                        save_checkpoint(accelerator, theta, iter, global_step, loss.item())
            theta_adapter = extract_lora_state_dict(theta, lora_config)
            old_adapter = extract_lora_state_dict(theta, lora_config)
        ref_adapter = extract_lora_state_dict(theta, lora_config)

    # ==== 4) 保存 ====
    if accelerator.is_main_process:
        save_checkpoint(accelerator, theta, iter, global_step, loss.item())
    print("Done.")


if __name__ == "__main__":
    main()