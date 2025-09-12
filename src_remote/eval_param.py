from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from PIL import Image
import torch
import json
from tqdm import tqdm
import shortuuid
import os
from collections import defaultdict
import json
import argparse
import nltk
from nltk.corpus import wordnet as wn

nltk.data.path.append('/nltk_data-gh-pages/nltk_data')

def are_synonyms(word1, word2):
    synsets1 = wn.synsets(word1)
    synsets2 = wn.synsets(word2)
    for synset1 in synsets1:
        for synset2 in synsets2:
            if synset1.path_similarity(synset2) is not None and synset1.path_similarity(synset2) > 0.8:
                return True
    return False

def evaluate_dataset(dataset_data, error_output_dir):
    all_categories = set(item['category'] for item in dataset_data)
    category_correct = {category: 0 for category in all_categories}
    category_incorrect = {category: 0 for category in all_categories}
    correct = 0
    incorrect = 0

    # 错误答案按类别分类存储
    errors_by_category = defaultdict(list)

    # 使用 tqdm 包裹数据集处理过程，显示进度条
    for item in dataset_data:
        gt = item['ground_truth'].lower()
        # 处理answer为None的情况
        answer = item['text'].split("assistant\n")[-1]
        if answer is None:
            answer = ""  # 将None转换为空字符串
        else:
            answer = answer.lower().rstrip('.')  # 移除答案末尾的句点
            
        category = item['category'].lower()
        
        if gt == answer:
            correct += 1
            category_correct[category] += 1
        else:
            if answer and are_synonyms(gt, answer):  # 确保answer不为空再检查同义词
                print(f'synonyms:{gt} and {answer}')
                correct += 1
                category_correct[category] += 1
            else:
                incorrect += 1
                category_incorrect[category] += 1
                # 将错误答案记录到字典中
                errors_by_category[category].append({
                    'question_id': item['question_id'],
                    'category': category,
                    'ground_truth': gt,
                    'answer': answer
                })

    # 打印结果
    print(f'Correct: {correct}')
    print(f'Incorrect: {incorrect}')
    print(f'Total: {correct + incorrect}')
    
    # 计算每个类别的精度
    percentage_list = []
    print("\nCategory-wise accuracies:")
    sorted_categories = sorted(category_correct.keys())
    
    overall_correct = 0
    overall_total = 0
    
    for cat in sorted_categories:
        cat_corr = category_correct[cat]
        cat_total = cat_corr + category_incorrect[cat]
        cat_acc = cat_corr / cat_total if cat_total > 0 else 0
        print(f"{cat}: {cat_corr}/{cat_total} ({cat_acc * 100:.2f}%)")
        percentage_list.append(cat_acc * 100)
        overall_correct += cat_corr
        overall_total += cat_total
    
    overall_acc = overall_correct / overall_total if overall_total > 0 else 0
    print(f"Overall Acc: {overall_acc*100:.2f}%")
    percentage_list.append(overall_acc * 100)

def evaluation_metrics(data_path, error_output_dir):
    # 读取数据
    with open(data_path, "r") as f:
        data = [json.loads(line) for line in f]
    
    # 按数据集分组
    datasets = defaultdict(list)
    for item in data:
        dataset_name = item['question_id'].split('_')[0]
        datasets[dataset_name].append(item)
    
    # 对每个数据集分别评估
    for dataset_name, dataset_data in datasets.items():
        print(f"\n{'='*50}")
        print(f"Evaluating dataset: {dataset_name}")
        print('='*50)
        evaluate_dataset(dataset_data, error_output_dir)

def eval_model(args):
    answers_file = args.results_file
    error_output_dir = args.error_output_dir  # 获取错误答案输出目录
    evaluation_metrics(answers_file, error_output_dir)

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
            image_name = q['image_name']
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
                        {"type": "text", "text": question_text},
                    ],
                },
            ]
            
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=[text], images=[image], return_tensors="pt").to(model.device)
            generate_ids = model.generate(**inputs, 
                                            max_new_tokens=128,
                                            do_sample=True,
                                            temperature=0.7,
                                            top_k=50)
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
    eval_test_folder()
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_file", type=str, default="./test_answers/test_answers.jsonl")
    parser.add_argument("--error_output_dir", type=str, default="./test_errors")  # 新增参数：输出目录
    args = parser.parse_args()
    eval_model(args=args)
    
