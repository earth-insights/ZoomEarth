import os
import json
import base64
from PIL import Image
from tqdm import tqdm
import shortuuid
from datasets import load_from_disk
from openai import AzureOpenAI
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor, as_completed
import re
import os
from openai import OpenAI
import argparse


Image.MAX_IMAGE_PIXELS = None

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="EMPTY"
)

deployment = "ZoomEarth"

instruction = """
You are an intelligent remote sensing analyst.
Given a natural language question about a satellite image, generate a structured reasoning answer as follows:
1. <think> ... </think>
    - Provide a neutral one-sentence description of the whole image scene.
    - Cropping task: "This question is asking about <short intent>, therefore I need to crop the image to examine the surroundings of the mentioned target."
    - Non-cropping task: "This question is asking about <short intent>, therefore I need to analyze the entire image without cropping."
    - Include:
        * Question Intent: describe the type of question (object category, spatial relation, count, etc.) and needed visual info.
        * Localization Strategy:
            - Cropping: approximate referent object location in natural language (no coordinates).
            - Non-cropping: strategy to detect all relevant objects.      * Reasoning Result:
    - Cropping: output exactly one JSON-formatted bbox for the referent:          [{"bbox_2d": [x_min,y_min,x_max,y_max], "label": "<short description>"}]
    - Non-cropping: summarize how detected objects will be used to produce the count.
2. <think> ... </think> (only when saw the cropped image)
    - Explain how to reason step by step from the referent (or detected objects) to the final answer. 
3. <answer> ... </answer>
    - Your final answer, use a single word or phrase.
Rules: 
    - Always return exactly one <answer> block, for tasks that need cropping, you can provide the bounding box of the object you are intrested, after given the cropped image, you can generate another <think> block to find the answer. 
    - For cropping tasks, also include a bounidng box in <stage_2_reasoning> block 
    - If unsure about localization, make a best guess—never say uncertain.
"""

def extract_bbox(completion_content: str, scale):
        pattern = r'"bbox_2d"\s*:\s*\[(.*?)\]'
        matches = re.findall(pattern, completion_content, re.DOTALL)

        bboxes = []
        for m in matches:
            try:
                nums = [float(x.strip()) for x in m.split(",")]
                bbox = [num * scale for num in nums]
                bboxes.append(bbox)
            except ValueError:
                continue
        return bboxes

def extract_answer(text):
    m = re.search(r'<answer>\s*(.*?)\s*</answer>', text)
    if not m:
        return None
    answer = m.group(1)
    return answer

def cut_image(image, bbox, min_size=512):
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
    if scale < 1:
        new_w = int(w * scale)
        new_h = int(h * scale)
        image = image.resize((new_w, new_h), Image.BICUBIC)
    return image, 1 / scale

def resize_image(image, max_size=512):
    w, h = image.size
    scale = max_size / max(w, h)
    if scale < 1:
        new_w = int(w * scale)
        new_h = int(h * scale)
        image = image.resize((new_w, new_h), Image.BICUBIC)
    return image

def encode_pil_image_to_data_url(image: Image.Image) -> str:
    buffered = BytesIO()
    format = 'PNG' if image.mode in ['RGBA', 'P'] else 'JPEG'
    image.save(buffered, format=format)
    mime_type = f"image/{format.lower()}"
    encoded = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"

def process_item(sample):
    image_fp = "./image/" + sample["image_name"].split("/")[-1]
    cur_prompt = sample["question"] + instruction

    try:
        image = Image.open(image_fp).convert("RGB")
        image_resized = resize_image(image)
        image_url = encode_pil_image_to_data_url(image_resized)

        messages_stage1 = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_url}},
                    {"type": "text", "text": cur_prompt},
                ],
            }
        ]

        response1 = client.chat.completions.create(
            model=deployment,
            messages=messages_stage1
        )
        output1 = response1.choices[0].message.content.strip()

    except Exception as e:
        return {
            "question_id": sample.get("question_id"),
            "ground_truth": sample["ground_truth"],
            "answer1": f"Error: {e}",
            "answer2": "",
            "bbox_ref": sample.get("bbox"),
            "bbox": "",
            "prompt": cur_prompt,
            "category": sample["category"],
            "stage1": f"Error: {e}",
            "stage2": "",
            "type": sample["type"],
            "image": sample["image_name"],
            "error": True,
            "model_id": "ZoomEarth (vllm)",
        }

    bbox = extract_bbox(output1)
    if bbox == []:
        return {
            "question_id": sample.get("question_id"),
            "ground_truth": sample["ground_truth"],
            "answer1": output1,
            "answer2": "",
            "bbox_ref": sample.get("bbox"),
            "bbox": [],
            "prompt": cur_prompt,
            "category": sample["category"],
            "stage1": output1,
            "stage2": "",
            "type": sample["type"],
            "image": sample["image_name"],
            "error": True,
            "model_id": "ZoomEarth (vllm)",
        }

    try:
        cropped = cut_image(image, bbox)
        cropped = resize_image(cropped)
        cropped_url = encode_pil_image_to_data_url(cropped)

        messages_stage2 = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_url}},
                    {"type": "text", "text": cur_prompt + instruction},
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": output1.split("<answer>")[0]},
                    {"type": "image_url", "image_url": {"url": cropped_url}},
                ]
            }
        ]

        response2 = client.chat.completions.create(
            model=deployment,
            messages=messages_stage2
        )
        output2 = response2.choices[0].message.content.strip()

    except Exception as e:
        output2 = f"Error: {e}"

    return {
        "question_id": sample.get("question_id"),
        "ground_truth": sample["ground_truth"],
        "answer1": output1,
        "answer2": output2,
        "bbox_ref": sample.get("bbox"),
        "bbox": bbox,
        "prompt": cur_prompt,
        "category": sample["category"],
        "stage1": output1,
        "stage2": output2,
        "type": sample["type"],
        "image": sample["image_name"],
        "error": False,
        "model_id": "ZoomEarth (vllm)"
    }

def eval_model_gpt_concurrent(max_workers=5, limit=10, exp_name="zoomearth_infer"):
    test_ds = load_from_disk("../LRS-GRO/test")

    test_subset = test_ds.select(range(min(limit, len(test_ds))))

    results = []
    os.makedirs("results", exist_ok=True)
    out_path = f"results/{exp_name}.jsonl"

    try:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(process_item, ex): ex for ex in test_subset}

            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
                result = future.result()
                results.append(result)
    finally:
        with open(out_path, "w", encoding="utf-8") as fout:
            for r in results:
                fout.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"✅ Done! {len(results)} samples generated at {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate LoRA model")
    parser.add_argument("--exp_name", type=str, required=True, help="Experiment name")
    args = parser.parse_args()
    eval_model_gpt_concurrent(max_workers=100, limit=3313, exp_name=args.exp_name)