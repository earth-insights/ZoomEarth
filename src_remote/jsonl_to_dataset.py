import json
import random
from datasets import Dataset, DatasetDict
import os

def main(
    input_jsonl_path="/home/zhangjunjie/CaoXiongyong_Student/fubowen/LRS_VQA/dataset/LRS_VQA_RL_V2.jsonl",
    output_dir="LRS_VQA_RL_V2",
    sft_count=1000,
    rl_count=5333,
    test_count=1000,
    seed=42
):
    # 1. 加载 jsonl
    with open(input_jsonl_path, "r", encoding="utf-8") as f:
        data = [json.loads(line.strip()) for line in f]

    print(f"总样本数: {len(data)}")
    total_required = sft_count + rl_count + test_count
    assert len(data) >= total_required, f"数据不足，至少需要 {total_required} 条样本"

    # 2. 打乱顺序
    random.seed(seed)
    random.shuffle(data)

    # 3. 切分
    test_data = data[:test_count]
    rl_data = data[test_count : test_count + rl_count]
    sft_data = data[test_count + rl_count:]

    print(f"SFT: {len(sft_data)}\nRL: {len(rl_data)}\nTEST: {len(test_data)}")

    # 4. 构建 DatasetDict
    dataset_dict = DatasetDict({
        "sft": Dataset.from_list(sft_data),
        "rl": Dataset.from_list(rl_data),
        "test": Dataset.from_list(test_data),
    })

    # 5. 保存到磁盘
    os.makedirs(output_dir, exist_ok=True)
    dataset_dict.save_to_disk(output_dir)
    print(f"✅ 数据集已保存到: {output_dir}")

if __name__ == "__main__":
    main()
