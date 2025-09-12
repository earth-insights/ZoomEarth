from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from PIL import Image
import torch
import json
from tqdm import tqdm
import shortuuid
import os

def cut_image(image, vertices, min_size=512):
    x1, y1, x2, y2 = map(int, vertices)
    center_x = (x1 + x2) // 2
    center_y = (y1 + y2) // 2

    width = x2 - x1
    height = y2 - y1

    # 如果宽或高小于min_size，则扩展为min_size，以中心为基准
    new_w = max(width, min_size)
    new_h = max(height, min_size)

    new_x1 = max(0, center_x - new_w // 2)
    new_y1 = max(0, center_y - new_h // 2)
    new_x2 = min(image.width, center_x + new_w // 2)
    new_y2 = min(image.height, center_y + new_h // 2)

    # 防止截出来不足min_size（例如在边缘），强制调整左上角
    if new_x2 - new_x1 < min_size and new_x2 == image.width:
        new_x1 = max(0, new_x2 - min_size)
    if new_y2 - new_y1 < min_size and new_y2 == image.height:
        new_y1 = max(0, new_y2 - min_size)

    return image.crop((new_x1, new_y1, new_x2, new_y2))
    

def resize_image(image, max_size=2048):
    w, h = image.size
    scale = max_size / max(w, h)
    if scale < 1:
        new_w = int(w * scale)
        new_h = int(h * scale)
        image = image.resize((new_w, new_h), Image.BICUBIC)
    return image

def downsample_image_4x(image):
    w, h = image.size
    new_w = w // 4
    new_h = h // 4
    image = image.resize((new_w, new_h), Image.BICUBIC)
    return image

def eval_model():
    Image.MAX_IMAGE_PIXELS = None

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-VL-7B-Instruct",
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True
    ).eval()

    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct", trust_remote_code=True)

    questions = []
    with open('./LRS_VQA/LRS_VQA_merged.jsonl', 'r', encoding='utf-8') as f:
        for line in f:
            questions.append(json.loads(line))

    answers_path = "./LRS_VQA/answers/answers_top_test.jsonl"
    os.makedirs(os.path.dirname(answers_path), exist_ok=True)
    with open(answers_path, "w", encoding="utf-8") as ans_file:
        for i, question in enumerate(tqdm(questions)):
            idx = question["question_id"]
            ground_truth = question['ground_truth']
            category = question['category']
            cur_prompt = question['text']
            image_path = question['image']
            
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
            image = Image.open(image_path).convert("RGB")
            
            if 'hbox' in question and question['hbox'] is not None:
                bbox = question['hbox']
            else:
                bbox = [0, 0, image.width, image.height]
            
            image = cut_image(image=image, vertices=bbox)
            image = resize_image(image)
            
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=[text], images=[image], return_tensors="pt").to(model.device)

            generate_ids = model.generate(**inputs, max_new_tokens=128)
            outputs = processor.tokenizer.batch_decode(generate_ids, skip_special_tokens=True)[0].strip()

            ans_id = shortuuid.uuid()
            ans_file.write(json.dumps({
                "question_id": idx,
                "prompt": cur_prompt,
                "category": category,
                "text": outputs,
                "ground_truth": ground_truth,
                "answer_id": ans_id,
                "model_id": "Qwen2-VL-7B-Instruct",
                "metadata": {}
            }) + "\n")

if __name__ == '__main__':
    eval_model()
