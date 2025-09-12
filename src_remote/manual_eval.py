from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from PIL import Image
import torch
import json
from tqdm import tqdm
import shortuuid
import os

def resize_image(image, max_size=1024):
    w, h = image.size
    scale = max_size / max(w, h)
    if scale < 1:
        new_w = int(w * scale)
        new_h = int(h * scale)
        image = image.resize((new_w, new_h), Image.BICUBIC)
    return image

def eval_test_folder(test_folder="test/", output_path="test_answers/test-answers.jsonl"):
    Image.MAX_IMAGE_PIXELS = None
    
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-VL-7B-Instruct",
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True
    ).eval()
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct", trust_remote_code=True)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    ans_file = open(output_path, "w", encoding="utf-8")
    
    # 遍历test文件夹下所有子文件夹
    for subfolder in os.listdir(test_folder):
        subfolder_path = os.path.join(test_folder, subfolder)
        if not os.path.isdir(subfolder_path):
            continue
        
        jsonl_file = None
        for f in os.listdir(subfolder_path):
            if f.endswith(".jsonl"):
                jsonl_file = os.path.join(subfolder_path, f)
                break
        if jsonl_file is None:
            print(f"跳过{subfolder_path}，未找到jsonl文件")
            continue
        
        print(f"正在处理：{subfolder_path}")
        
        # 加载问题
        questions = []
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line in f:
                questions.append(json.loads(line))
        
        for idx, q in enumerate(tqdm(questions, desc=subfolder)):
            image_name = q['image'].split('/')[-1]
            # image_name = q['image_name']
            question_text = q['question']
            question_id = q['question_id']
            category = q.get('category', '')
            ground_truth = q.get('ground_truth', '')
            
            # 匹配图片：假设图片和jsonl文件在同一文件夹下
            image_file = os.path.join(subfolder_path, f"{image_name.split('.')[0]}_{idx+1}.png")
            if not os.path.exists(image_file):
                print(f"未找到图片 {image_file}，跳过该问题。")
                continue
            
            image = Image.open(image_file).convert("RGB")
            image = resize_image(image)
            
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": question_text+"\nChoose the most appropriate answer from the above four options and just reply with one of the letters A, B, C or D."},
                        # {"type": "text", "text": question_text},
                    ],
                },
            ]
            
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=[text], images=[image], return_tensors="pt").to(model.device)
            generate_ids = model.generate(**inputs, 
                                            max_new_tokens=128,
                                            do_sample=True,
                                            temperature=0.3,
                                            top_k=5)
            outputs = processor.tokenizer.batch_decode(generate_ids, skip_special_tokens=True)[0].strip()
            
            answer_id = shortuuid.uuid()
            ans_file.write(json.dumps({
                "question_id": question_id,
                "prompt": question_text,
                "category": category,
                "text": outputs,
                "ground_truth": ground_truth,
                "answer_id": answer_id,
                "model_id": "Qwen2-VL-7B-Instruct",
                "metadata": {}
            }, ensure_ascii=False) + "\n")
    
    ans_file.close()
    print(f"所有子文件夹处理完成，答案保存在 {output_path}")

if __name__ == "__main__":
    eval_test_folder("xlrs", "xlrs-test_answers/test-answers.jsonl")
