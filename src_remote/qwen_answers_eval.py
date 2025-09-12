import argparse
import json
from tqdm import tqdm
from collections import defaultdict
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
    all_categories = set(item['category'].lower() for item in dataset_data)
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
    
    # 打印纯数字的百分比值
    print("\nPercentages:")
    for p in percentage_list:
        s = f"{p:.2f}".rstrip('0').rstrip('.')
        print(s)

    # 将错误答案按类别分类保存到不同的 JSONL 文件
    print(f"\nSaving incorrect answers to individual JSONL files...")
    for category, errors in errors_by_category.items():
        # 为每个类别创建一个文件
        error_file_path = f"{error_output_dir}/{category}_incorrect_answers.jsonl"
        with open(error_file_path, 'w') as f:
            for error in errors:
                f.write(json.dumps(error) + '\n')

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_file", type=str, default="/home/zhangjunjie/CaoXiongyong_Student/fubowen/LRS_VQA/results/lora_infer_zero_shot.jsonl")
    parser.add_argument("--error_output_dir", type=str, default="./LRS_VQA/errors_3B")  # 新增参数：输出目录
    args = parser.parse_args()
    
    eval_model(args)
