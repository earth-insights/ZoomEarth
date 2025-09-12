import os
import json
import base64
from tqdm import tqdm
from openai import OpenAI
from PIL import Image
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import AzureOpenAI

endpoint = os.getenv("ENDPOINT_URL", "https://drivevla-new.openai.azure.com/")
deployment = os.getenv("DEPLOYMENT_NAME", "gpt-4o")
subscription_key = os.getenv(
    "AZURE_OPENAI_API_KEY",
    "BFTb9qdU0YCqpgPCdep2AgdjEdv8nkf9zlJa2sBNgYXwB968O6ncJQQJ99BDACHYHv6XJ3w3AAABACOG0sJL",
)
Image.MAX_IMAGE_PIXELS = None

# 使用 OpenAI SDK 兼容 DashScope API
client = AzureOpenAI(
    azure_endpoint=endpoint,
    api_key=subscription_key,
    api_version="2024-05-01-preview",
)

def load_image_from_path(image_path: str) -> Image.Image:
    """
    从给定路径加载一张图片为 PIL Image 实例。
    """
    from PIL import Image
    return Image.open(image_path).convert("RGB")  # 推荐转为RGB避免模式问题

def resize_image(image, max_size=1024):
    """缩放图像到最大边为max_size，保持比例"""
    w, h = image.size
    scale = max_size / max(w, h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    return image.resize((new_w, new_h), Image.BICUBIC)

def encode_pil_image_to_data_url(image: Image.Image) -> str:
    """
    将 PIL Image 编码为 base64 的 data URL 字符串。
    """
    buffered = BytesIO()
    format = 'PNG' if image.mode in ['RGBA', 'P'] else 'JPEG'
    image.save(buffered, format=format)
    mime_type = f"image/{format.lower()}"
    encoded = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded}"

def process_item(item, image_dir, model_name="qwen2-vl-72b-instruct"):
    # result = {
    #     "question_id": item.get("question_id"),
    #     "image_name": item.get("image_name"),
    #     "category": item.get("category"),
    #     "question": item.get("question"),
    #     "ground_truth": item.get("ground_truth"),
    #     "cut": item.get("cut"),
    #     "global": item.get("global"),
    #     "stage_1_reasoning": "",
    #     "stage_2_reasoning": "",
    #     "area": "",
    #     "bbox": [],
    #     "scale": 1
    # }
    result = item

    if item['cut']:
        return result
    
    image_path = os.path.join(image_dir, result["image_name"])
    image = load_image_from_path(image_path)
    image = resize_image(image)
    image_url = encode_pil_image_to_data_url(image)

    try:
        prompt = f"""
You are an intelligent remote sensing analyst.  
Your task is to generate reasoning-based annotations for Visual Question Answering (VQA) using satellite imagery.  

I will provide:  
- A global satellite image (downsampled for efficiency)  
- A natural language question referring to the image  
- The ground truth answer  

Important:  
- For this type of question, the reasoning must be performed on the **entire global image** without cropping.  
- All provided questions are **counting tasks** (e.g., number of bridges, number of buildings).  
- The final answer should be derived by identifying and counting all relevant objects visible in the global image.  

When writing <stage_1_reasoning>, follow these rules:  
  * At the very start of `<stage_1_reasoning>`, always begin with a single sentence in English of the following form — replacing `<short intent>` with a concise summary of the question intent:

    This question is asking about <short intent>, therefore I need to analyze the entire image without cropping.

  * In **Localization Strategy**, describe how to locate all potential instances of the target object type across the global image (e.g., "look for long narrow structures crossing the river for bridges").  
  * In **Reasoning Result**, describe how the identified objects will be used to produce the final count. Do not reveal the ground truth number here.  

Your output must strictly follow this structure:

<global> - Provide a brief but informative description of the global satellite image (e.g., main structures, spatial layout). <\global>
<stage_1_reasoning>

Question Intent: Identify the type of question being asked (e.g., object count), and determine what kind of visual information is needed to answer it.

Localization Strategy: Parse the question to identify the target object type (e.g., bridges, roads, buildings). Explain how to visually detect these objects across the global image, using cues such as shape, size, color, and spatial arrangement.

Reasoning Result: Summarize how the detected objects will be counted to produce the final answer. Do not state the answer itself in this section.
<\stage_1_reasoning>

<stage_2_reasoning>

Given the identified objects in the global image, explain how to reason step by step to determine the final count.  
Clearly connect the detection of individual objects to the overall counting process.  
<\stage_2_reasoning>

Constraints:

Do not reveal the final answer in <stage_1_reasoning>.

The <global> description must be neutral and avoid giving away the answer.

The <stage_2_reasoning> must directly connect the detection of objects to the final count.
 
Input:

Question: {result["text"]}

Ground Truth Answer: {result["ground_truth"]}
"""

        messages = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text":
                        "You are an expert in visual reasoning over satellite images. Given one image and a counting-type question, your task is to explain your reasoning step by step. \
                    First, describe the global image. Then explain why cropping is unnecessary for this type of question and how you will systematically scan the entire image to find all relevant objects. \
                    Finally, describe the identification rules, counting process, and quality checks you would apply. \
                    You must not provide the final count itself — only the reasoning steps, structured in the requested sections."
                    }
                ]
            },
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_url}},
                    {"type": "text", "text": prompt}
                ]
            }
        ]

        response = client.chat.completions.create(
            model=deployment,
            messages=messages,
            max_tokens=4096,
            temperature=0.6,
            top_p=0.95,
            frequency_penalty=0,
            presence_penalty=0,
            stop=None,
            stream=False,
        )
        answer = response.choices[0].message.content

        result.update({
            "stage_1_reasoning": answer.split("<stage_1_reasoning>")[-1].split("<\\stage_1_reasoning>")[0],
            "stage_2_reasoning": answer.split("<stage_2_reasoning>")[-1].split("<\\stage_2_reasoning>")[0],
            "cut": False,
            "global": answer.split("<global>")[-1].split("<\\global>")[0],
        })
    except Exception as e:
        result["cut"] = True
        result["stage_1_reasoning"] = f"Error: {str(e)}"

    return result

# def main(image_dir, jsonl_in, jsonl_out, model_name="qwen2-vl-72b-instruct", max_workers=50):
#     with open(jsonl_in, 'r', encoding='utf-8') as fin:
#         items = [json.loads(line) for line in fin]

#     results = []
#     with ThreadPoolExecutor(max_workers=max_workers) as executor:
#         futures = [
#             executor.submit(process_item, item, image_dir, model_name)
#             for item in items
#         ]
#         for future in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
#             results.append(future.result())

#     with open(jsonl_out, 'w', encoding='utf-8') as fout:
#         for r in results:
#             fout.write(json.dumps(r, ensure_ascii=False) + "\n")

def main(image_dir, jsonl_in, jsonl_out, model_name="qwen2-vl-72b-instruct", max_workers=50):
    with open(jsonl_in, 'r', encoding='utf-8') as fin:
        items = [json.loads(line) for line in fin]

    with ThreadPoolExecutor(max_workers=max_workers) as executor, \
        open(jsonl_out, 'w', encoding='utf-8') as fout:

        futures = {executor.submit(process_item, item, image_dir, model_name): item for item in items}

        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
            try:
                r = future.result(timeout=300)  # 给每个 future 一个最大超时
                fout.write(json.dumps(r, ensure_ascii=False) + "\n")
                fout.flush()
            except Exception as e:
                bad_item = futures[future]
                print(f"Error on item: {bad_item.get('id', '')}, error: {e}")

if __name__ == "__main__":
    main("LRS_VQA/image", "dataset/vl_answers.jsonl", "dataset/vl_answers_fixed.jsonl")
