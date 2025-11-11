import json
from PIL import Image
import nltk
import os
import re
import numpy as np
import torch
from peft import PeftModel
from transformers import Qwen2VLProcessor, Qwen2VLForConditionalGeneration
from datetime import datetime

TEMPERATURE = 0.7
MAX_TOKENS = 1024

# TODO: change to your own corpora path 
local_corpora = 'LOCAL_COPRA'
nltk.data.path.insert(0, local_corpora)
from nltk.corpus import wordnet as wn
Image.MAX_IMAGE_PIXELS = None

def extract_bboxes(completion_content: str):
    pattern = r'"bbox_2d"\s*:\s*\[(.*?)\]'
    matches = re.findall(pattern, completion_content, re.DOTALL)

    bboxes = []
    for m in matches:
        try:
            nums = [float(x.strip()) for x in m.split(",")]
            bboxes.append(nums)
        except ValueError:
            continue
    return bboxes

def tanh(x):
    return 2 / (1 + np.exp(-2 * x)) - 1

def cut_image(image, bbox, min_size=512):
    if len(bbox) != 4:
        return image
    x1, y1, x2, y2 = map(int, bbox)
    width, height = x2 - x1, y2 - y1

    if width < min_size or height < min_size:
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2

        new_x1 = center_x - min_size // 2
        new_y1 = center_y - min_size // 2
        new_x2 = new_x1 + min_size
        new_y2 = new_y1 + min_size

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

        new_x1 = max(0, new_x1)
        new_y1 = max(0, new_y1)
        new_x2 = min(image.width, new_x1 + min_size)
        new_y2 = min(image.height, new_y1 + min_size)

        return image.crop((int(new_x1), int(new_y1), int(new_x2), int(new_y2)))

    else:
        cropped = image.crop((x1, y1, x2, y2))
        return cropped

def resize_image(image, max_size=512):
    w, h = image.size
    scale = max_size / max(w, h)
    min_scale = 30 / min(w,h)
    scale = max(min_scale, scale)
    if scale < 1:
        new_w = int(w * scale)
        new_h = int(h * scale)
        image = image.resize((new_w, new_h), Image.BICUBIC)
    return image

def _lemmatize(word:str) -> str:
    try:
        from nltk.stem import WordNetLemmatizer
        return WordNetLemmatizer().lemmatize(word)
    except Exception:
        return word

def synonyms_degree(word1:str, word2:str):
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
    return best if best < 0.8 else 1

def correctness(answer:str, gt:str):
    if answer is None:
        answer = ""
    answer = answer.strip().lower().rstrip('.')
    gt = (gt or "").strip().lower().rstrip('.')
    if not gt:
        return 0
    if answer == gt:
        return 1
    else:
        return synonyms_degree(gt, answer)

def extract_tag(text: str, tag: str, default=None):
    safe = re.escape(tag)
    m = re.compile(rf'<{safe}\s*>\s*(.*?)\s*</{safe}\s*>', re.S).search(text or "")
    return m.group(1).strip() if m else default

def parse_bbox_text(s):
    if s is None:
        return None
    try:
        arr = json.loads(s)
        if isinstance(arr, list) and len(arr) == 4:
            return [float(v) for v in arr]
    except Exception:
        pass
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
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2

        new_x1 = center_x - min_size // 2
        new_y1 = center_y - min_size // 2
        new_x2 = new_x1 + min_size
        new_y2 = new_y1 + min_size

        new_x1 = max(0, new_x1)
        new_y1 = max(0, new_y1)

        return [int(new_x1), int(new_y1), int(new_x2), int(new_y2)]
    else:
        return bbox

def iou(box_a, box_b):
    if len(box_b)!=4 or len(box_a)!=4:
        return 0
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

def chat_batch(prompts, imgs, processor, accelerator, model):
    inputs = processor(
        text=prompts,
        images=imgs,
        return_tensors="pt",
        padding="longest",
        text_pair=["" for _ in prompts]
    ).to(accelerator.device)
    if "labels" in inputs:
        inputs.pop("labels")

    gen_ids = accelerator.unwrap_model(model).generate(
        **inputs,
        max_new_tokens=800,
        do_sample=True,
        num_beams=1,
        temperature=TEMPERATURE,
    )

    return gen_ids, inputs

def chat(prompt, imgs, processor, accelerator, model):
    inputs = processor(
        text=[prompt],
        images=imgs,
        return_tensors="pt",
        padding="longest",
        text_pair=[""]
    ).to(accelerator.device)
    if "labels" in inputs:
        inputs.pop("labels")

    gen_ids = accelerator.unwrap_model(model).generate(
        **inputs,
        max_new_tokens=800,
        do_sample=True,
        num_beams=1,
        temperature=TEMPERATURE,
    )

    return gen_ids, inputs


# format reward
def get_format_reward(completion1, completion2, **kwargs):
    rewards = []
    for i in range(len(completion1)):
        reward = get_format_reward_item(completion1[i], completion2[i], i, **kwargs)
        rewards.append(reward)
    return rewards

def get_format_reward_item(completion1, completion2, idx, **kwargs):
    cut = completion2 != ""
    format_reward = 0.0
    if cut:
        bbox = extract_bboxes(completion1)
        if extract_tag(completion1, "think") and extract_tag(completion2, "think") and extract_tag(completion2, "answer") and bbox != []:
            format_reward = 1
        else:
            format_reward = 0
    else:
        if extract_tag(completion1, "think") and extract_tag(completion1, "answer"):
            format_reward = 1
        else:
            format_reward = 0 
    if os.getenv("DEBUG_MODE") == "true":
        log_path = os.getenv("LOG_PATH")
        current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
        image_path = kwargs.get("image_path")[idx] if "image_path" in kwargs else None
        problem = kwargs.get("question")[idx]
        if format_reward <=1.0:  # this condition can be changed for debug
            with open(log_path+"_format.txt", "a", encoding='utf-8') as f:
                f.write(f"------------- {current_time} format reward: {format_reward} -------------\n")
                f.write(f"image_path: {image_path}\n")
                f.write(f"problem: {problem}\n")
                f.write(f"Completion1: {completion1}\n")
                f.write(f"Completion2: {completion2}\n")
    return format_reward

# iou reward + r-g reward
def get_bbox_reward(completion1, **kwargs):
    rewards = []
    for i in range(len(completion1)):
        reward = get_bbox_reward_item(completion1[i] , i, **kwargs)
        rewards.append(reward)
    return rewards

def get_bbox_reward_item(completion, idx, **kwargs):   
    def extract_bboxes(completion_content: str):
        pattern = r'"bbox_2d"\s*:\s*\[(.*?)\]'
        matches = re.findall(pattern, completion_content, re.DOTALL)

        bboxes = []
        for m in matches:
            try:
                nums = [float(x.strip()) for x in m.split(",")]
                bboxes.append(nums)
            except ValueError:
                continue
        return bboxes
    bbox_reward = 0.0
    bbox = extract_bboxes(completion)
    if bbox != []:
        bbox = bbox[0]
        bbox_ref = kwargs.get("bbox")[idx]
        if len(bbox) != 4 or len(bbox_ref) != 4:
            bbox_reward = 0
        elif bbox == bbox_ref:
            bbox_reward = 2
        else:
            bbox_ref = [point * kwargs.get("scale")[idx] / 2 for point in kwargs.get("bbox")[idx]]

            bbox = [kwargs['scale'][idx] * point for point in bbox]
            
            if isinstance(bbox_ref, (list, tuple)) and len(bbox_ref) == 4 and isinstance(bbox, (list, tuple)) and len(bbox) == 4:
                cx, cy = (bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0
                rx, ry = (bbox_ref[0] + bbox_ref[2]) / 2.0, (bbox_ref[1] + bbox_ref[3]) / 2.0
                distance = ((rx - cx) ** 2 + (ry - cy) ** 2) ** 0.5 + 1e-6
                bbox_iou = iou(bbox, bbox_ref)
                # bbox_reward = float(bbox_iou + tanh(50.0 / distance))
                bbox_reward = float(bbox_iou + tanh(200 / distance))
    if os.getenv("DEBUG_MODE") == "true":
        log_path = os.getenv("LOG_PATH")
        current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
        image_path = kwargs.get("image_path")[idx] if "image_path" in kwargs else None
        problem = kwargs.get("question")[idx]
        with open(log_path+"_bbox.txt", "a", encoding='utf-8') as f:
            f.write(f"------------- {current_time} Bbox reward: {bbox_reward} -------------\n")
            f.write(f"image_path: {image_path}\n")
            f.write(f"problem: {problem}\n")
            f.write(f"Completion: {completion}\n")
            f.write(f"BBox: {bbox}\n")
            if kwargs.get("bbox")[idx] != []:
                f.write(f"Solution: {[point * kwargs.get('scale')[idx] / 2 for point in kwargs.get('bbox')[idx]]}\n")
            else:
                f.write("Solution: []\n")
    return bbox_reward

# answer reward
def get_answer_reward(completion1, completion2, **kwargs):
    rewards = []
    for i in range(len(completion1)):
        reward = get_answer_reward_item(completion1[i], completion2[i], i, **kwargs)
        rewards.append(reward)
    return rewards

def get_answer_reward_item(completion1, completion2, idx, **kwargs):
    cut = completion2 != ""
    answer = extract_tag(completion1, "answer")
    answer_ref = kwargs.get("ground_truth")[idx]
    answer_reward = 0
    if not cut:
        answer_reward = correctness(answer, answer_ref)
    else:
        answer = extract_tag(completion2, "answer")
        answer_reward = correctness(answer, answer_ref)
    if os.getenv("DEBUG_MODE") == "true":
        log_path = os.getenv("LOG_PATH")
        current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
        image_path = kwargs.get("image_path")[idx] if "image_path" in kwargs else None
        problem = kwargs.get("question")[idx]
        with open(log_path+"_answer.txt", "a", encoding='utf-8') as f:
            f.write(f"------------- {current_time} answer reward: {answer_reward if cut else answer_reward / 3} -------------\n")
            f.write(f"image_path: {image_path}\n")
            f.write(f"problem: {problem}\n")
            f.write(f"Completion1: {completion1}\n")
            f.write(f"Completion2: {completion2}\n")
            f.write(f"Solution: {kwargs['ground_truth'][idx]}\n")  
    return answer_reward