import argparse
# import torch
import os
import json
from tqdm import tqdm
# import shortuuid
from PIL import Image
import math
import requests
from io import BytesIO
import re
# import torch.nn.functional as F
import copy
from collections import defaultdict
Image.MAX_IMAGE_PIXELS = 10000000000
import nltk

# TODO: add local corpora path
local_corpora = ''
nltk.data.path.insert(0, local_corpora)
from nltk.corpus import wordnet as wn

def _lemmatize(word:str) -> str:
    try:
        from nltk.stem import WordNetLemmatizer
        return WordNetLemmatizer().lemmatize(word)
    except Exception:
        return word

def are_synonyms(word1:str, word2:str) -> bool:
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

def evaluate_dataset(base):
    total_correct1 = 0
    total_correct2 = 0
    total_samples = len(base)
    
    total_correct_call = 0

    fixed_cases = []
    wrong_cases = []
    
    type_correct1 = defaultdict(int)
    type_correct2 = defaultdict(int)
    type_total = defaultdict(int)
    
    print("Processing evaluations...")
    for item in tqdm(base):
        category = item.get('category')
        gt = item.get('ground_truth', '').lower()
        answer1 = item.get('answer1', '')
        answer2 = item.get('answer2', '')
        if not answer1 == None:
            answer1 = answer1.lower().strip()
        if not answer2 == None:
            answer2 = answer2.lower().strip()
        else:
            answer2 = answer1

        is_correct1 = False
        is_correct2 = False
        if gt == answer1 or are_synonyms(gt, answer1):
            is_correct1 = True
        
        if gt == answer2 or are_synonyms(gt, answer2):
            is_correct2 = True

        if is_correct1 and not is_correct2:
            wrong_cases.append(item)
        if not is_correct1 and is_correct2:
            fixed_cases.append(item)

        if is_correct1:
            total_correct1 += 1
        if is_correct2:
            total_correct2 += 1
        
        type_name = item['type']
        type_total[type_name] += 1
        if is_correct1:
            type_correct1[type_name] += 1
        if is_correct2:
            type_correct2[type_name] += 1

    print("\n--- Evaluation Results ---")
    print(f'Total Correct (stage 1): {total_correct1}')
    print(f'Total Correct (stage 2): {total_correct2}')
    print(f'Total Incorrect (stage 1): {total_samples - total_correct1}')
    print(f'Total Incorrect (stage 2): {total_samples - total_correct2}')
    print(f'Total Samples: {total_samples}')
    print("-" * 25)

    print("Category-wise Accuracies:")
    
    print("-" * 25)

    print("Type-wise Accuracies:")
    for t in sorted(type_total.keys()):
        if type_total[t] > 0:
            acc1 = type_correct1[t] / type_total[t]
            acc2 = type_correct2[t] / type_total[t]
            print(f"{t:<15}: {acc1 * 100:.2f}% -> {acc2 * 100:.2f}%")
        else:
            print(f"{t:<15}: 0/0 (N/A)")
    
    print("-" * 25)

    if total_samples > 0:
        overall_acc_oa1 = total_correct1 / total_samples
        print(f"Overall Accuracy (OA, stage 1): {overall_acc_oa1 * 100:.2f}%")
        overall_acc_oa2 = total_correct2 / total_samples
        print(f"Overall Accuracy (OA, stage 2): {overall_acc_oa2 * 100:.2f}%")
        tool_use_accuracy = total_correct_call / total_samples
        print(f"Tool use accuracy: {tool_use_accuracy * 100:.4f}%")
    else:
        print("Overall Accuracy (OA): N/A (No samples found)")
    return total_correct1, total_correct2, total_samples



def evaluation_metrics(data_path):
    total_correct1, total_correct2, total_samples = 0, 0, 0
    with open(data_path, "r") as f:
        data = [json.loads(line) for line in f]
    
    datasets = defaultdict(list)
    for item in data:
        dataset_name = "LRS-GRO"
        datasets[dataset_name].append(item)
    
    for dataset_name, dataset_data in datasets.items():
        print(f"\n{'='*50}")
        print(f"Evaluating dataset: {dataset_name}")
        print('='*50)
        temp1, temp2, temp3 = evaluate_dataset(dataset_data)
        total_correct1 += temp1
        total_correct2 += temp2
        total_samples += temp3
        cat_acc1 = total_correct1 / total_samples
        cat_acc2 = total_correct2 / total_samples
    print(f"Overall: {cat_acc1 * 100:.2f}% -> {cat_acc2 * 100:.2f}%")
        

def eval_model(args):
    answers_file = args.results_file
    evaluation_metrics(answers_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_file", type=str, default="")
    args = parser.parse_args()
    
    eval_model(args)