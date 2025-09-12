from transformers import AutoProcessor, Qwen2VLForConditionalGeneration, Qwen2_5_VLForConditionalGeneration
from PIL import Image
import torch
import json
from tqdm import tqdm
import shortuuid
import os

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

    # model = Qwen2VLForConditionalGeneration.from_pretrained(
    #     "Qwen/Qwen2-VL-7B-Instruct",
    #     device_map="auto",
    #     torch_dtype=torch.float16,
    #     trust_remote_code=True
    # ).eval()

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-3B-Instruct",
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True
    ).eval()

    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct", trust_remote_code=True)

    questions = []
    with open('./LRS_VQA/LRS_VQA_merged.jsonl', 'r', encoding='utf-8') as f:
        for line in f:
            questions.append(json.loads(line))

    answers_path = "./LRS_VQA/answers/answers.jsonl"
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
