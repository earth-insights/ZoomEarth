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
# deployment = os.getenv("DEPLOYMENT_NAME", "gpt-4o")
deployment = os.getenv("DEPLOYMENT_NAME", "gpt-5-chat")
subscription_key = os.getenv(
    "AZURE_OPENAI_API_KEY",
    "BFTb9qdU0YCqpgPCdep2AgdjEdv8nkf9zlJa2sBNgYXwB968O6ncJQQJ99BDACHYHv6XJ3w3AAABACOG0sJL",
)


# 使用 OpenAI SDK 兼容 DashScope API
client = AzureOpenAI(
    azure_endpoint=endpoint,
    api_key=subscription_key,
    api_version="2024-05-01-preview",
)

Image.MAX_IMAGE_PIXELS = None

def cut_image(image, bbox, min_size=512):
    """
    最终输出统一为 min_size × min_size：
    - 若 bbox 边长 < min_size → 在原图中扩展，必要时平移避免越界；
    - 若 bbox 边长 >= min_size → 对 bbox 区域缩放并中心裁剪。
    """
    x1, y1, x2, y2 = map(int, bbox)
    width, height = x2 - x1, y2 - y1

    if width < min_size or height < min_size:
        # 中心点
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2

        # 初步计算边界
        new_x1 = center_x - min_size // 2
        new_y1 = center_y - min_size // 2
        new_x2 = new_x1 + min_size
        new_y2 = new_y1 + min_size

        # 平移使得裁剪框在图像内部
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

        # 最后确保框不越界
        new_x1 = max(0, new_x1)
        new_y1 = max(0, new_y1)
        new_x2 = min(image.width, new_x1 + min_size)
        new_y2 = min(image.height, new_y1 + min_size)

        return image.crop((int(new_x1), int(new_y1), int(new_x2), int(new_y2)))

    else:
        # 普通中心缩放逻辑
        cropped = image.crop((x1, y1, x2, y2))
        w, h = cropped.size
        scale = min_size / min(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cropped.resize((new_w, new_h), Image.BICUBIC)

        left = (new_w - min_size) // 2
        top = (new_h - min_size) // 2
        return resized.crop((left, top, left + min_size, top + min_size))

def resize_image(image, max_size=1024):
    """缩放图像到最大边为max_size，保持比例"""
    w, h = image.size
    scale = max_size / max(w, h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    return image.resize((new_w, new_h), Image.BICUBIC)

# def encode_image_to_data_url(image_path):
#     with open(image_path, "rb") as f:
#         encoded = base64.b64encode(f.read()).decode("utf-8")
#     mime_type = "image/png" if image_path.lower().endswith(".png") else "image/jpeg"
#     return f"data:{mime_type};base64,{encoded}"

def load_image_from_path(image_path: str) -> Image.Image:
    """
    从给定路径加载一张图片为 PIL Image 实例。
    """
    from PIL import Image
    return Image.open(image_path).convert("RGB")  # 推荐转为RGB避免模式问题

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

def process_cut_item(item, image_dir, model_name="qwen2-vl-72b-instruct"):
    result = {
        "question_id": item.get("question_id"),
        "image_name": item.get("image").split("/")[-1],
        "category": item.get("category"),
        "question": item.get("text"),
        "ground_truth": item.get("ground_truth"),
        "cut": False,
        "global": "",
        "stage_1_reasoning": "",
        "stage_2_reasoning": "",
        "bbox": [],
        "scale": 1
    }

    image_path = os.path.join(image_dir, result["image_name"])
    if not os.path.exists(image_path):
        return result

    hbox = item.get("hbox")
    if not hbox or len(hbox) != 4:
        return result
    
    try:
        image1 = load_image_from_path(image_path)
        w, h = image1.size
        scale = max(max(w, h) / 1024, 1)

        image2 = cut_image(image1, hbox)
        image1 = resize_image(image1)

        image_url_1 = encode_pil_image_to_data_url(image1)
        image_url_2 = encode_pil_image_to_data_url(image2)
        prompt = f"""
You are an intelligent remote sensing analyst.  
Your task is to generate reasoning-based annotations for Visual Question Answering (VQA) using satellite imagery.  

I will provide:  
- A global satellite image (downsampled for efficiency)  
- The bounding box [x_min, y_min, x_max, y_max] of the **reference object** mentioned in the question  
- A natural language question referring to the image  
- The ground truth answer  

Important:  
- The reference bounding box corresponds to the **referent object** in the question (e.g., if the question asks "What is the structure parallel and closest to the bridge?", the reference bbox is for the bridge).  
- The final target answer is derived **by reasoning relative to this referent object**.  
- The global image should only be used for context and to explain how one would locate the referent region.  
- The final answer must be derived **by analyzing the cropped region corresponding to the referent and its surroundings**.  

When writing <stage_1_reasoning>, follow these rules:  
  * At the very start of `<stage_1_reasoning>`, always begin with a single sentence in English of the following form — replacing `<short intent>` with a concise summary of the question intent (no coordinates):
    This question is asking about <short intent>, therefore I need to crop the image to examine the surroundings of the mentioned target.
  * In **Localization Strategy**, describe the approximate location of the referent object in natural language (e.g., "the bottom-most long bridge across the river"). Do not output exact coordinates here.  
  * In **Reasoning Result**, first output exactly the line:  
    I need to pay attention to the reference object at  
    Immediately after that line, output the bounding box in the following JSON format, and nothing else in this section:  

```json
[
    {{"bbox_2d": [x_min, y_min, x_max, y_max], "label": "<short description of the referent object>"}}
]```

Your output must strictly follow this structure:

<global> - Provide a brief but informative description of the global satellite image (e.g., main structures, spatial layout). <\global>
<stage_1_reasoning>

Question Intent: Identify the type of question being asked (e.g., object category, count, color, spatial relation, etc.), and determine what kind of visual information is needed to answer it.

Localization Strategy: Parse the question to identify the **referent object** (e.g., bridge, river, building cluster). Translate the description into a visual query and locate it in the global image using semantic cues (shape, size, color, spatial arrangement). Summarize the approximate location of the referent object in natural language (e.g., "the wide horizontal bridge in the lower part of the image"). Do NOT put coordinates here.

Reasoning Result: First write the single line:
I need to pay attention to the reference object at
Then on the next line output only the JSON-formatted bounding box (as shown above) and nothing else. Do not include any additional explanation in this field.
<\stage_1_reasoning>

<stage_2_reasoning>

Given the cropped region of the **referent object**, explain how to reason about the final target answer.  
Specify what visual features or spatial relations should be observed (e.g., "look for structures parallel and adjacent to the bridge", "check which object is closest to the referent").  

Clearly connect the reasoning steps from the referent object to the final answer.  
<\stage_2_reasoning>

Constraints:

Do not reveal the final answer in <stage_1_reasoning>.

The <global> description must be neutral and avoid giving away the answer.

The <stage_2_reasoning> must directly connect the referent to the final target.
 
Input:

Question: {result["question"]}

Ground Truth Answer: {result["ground_truth"]}

Reference Bounding box: {[int(x / scale) for x in hbox]}
"""

        messages = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": 
                            "You are an expert in visual reasoning over satellite images. Given one or more images and a question, your task is to explain your reasoning on how to locate the target and answer the question. You must not provide the answer itself — only your reasoning steps."
                    }
                ]
            },
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_url_1}},
                    {"type": "image_url", "image_url": {"url": image_url_2}},
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
            "cut": True,
            "global": answer.split("<global>")[-1].split("<\\global>")[0],
            "bbox": [int(x / scale) for x in hbox],
            "scale": scale
        })
    except Exception as e:
        result["cut"] = True
        result["stage_1_reasoning"] = f"Error: {str(e)}"

    return result

def process_uncut_item(item, image_dir, model_name="qwen2-vl-72b-instruct"):
    result = {
        "question_id": item.get("question_id"),
        "image_name": item.get("image").split("/")[-1],
        "category": item.get("category"),
        "question": item.get("text"),
        "ground_truth": item.get("ground_truth"),
        "cut": False,
        "global": "",
        "stage_1_reasoning": "",
        "stage_2_reasoning": "",
        "bbox": [],
        "scale": 1
    }

    image_path = os.path.join(image_dir, result["image_name"])
    if not os.path.exists(image_path):
        return result

    try:
        image = load_image_from_path(image_path)
        image = resize_image(image)

        image_url = encode_pil_image_to_data_url(image)
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

Question: {result["question"]}

Ground Truth Answer: {result["ground_truth"]}
"""


        messages = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": 
                            "You are an expert in visual reasoning over satellite images. Given one or more images and a question, your task is to explain your reasoning on how to locate the target and answer the question. You must not provide the answer itself — only your reasoning steps."
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
            "global": answer.split("<global>")[-1].split("<\\global>")[0],
        })
    except Exception as e:
        result["cut"] = False
        result["stage_1_reasoning"] = f"Error: {e}"

    return result

def process_item(item, image_dir, model_name="qwen2-vl-72b-instruct"):
    hbox = item.get("hbox")
    if not hbox or len(hbox) != 4:
        return process_uncut_item(item, image_dir, model_name)
    else:
        return process_cut_item(item, image_dir, model_name)

def main(image_dir, jsonl_in, jsonl_out, model_name="qwen2-vl-72b-instruct", max_workers=100):
    with open(jsonl_in, 'r', encoding='utf-8') as fin:
        items = [json.loads(line) for line in fin]

    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(process_item, item, image_dir, model_name)
            for item in items
        ]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
            results.append(future.result())

    with open(jsonl_out, 'a', encoding='utf-8') as fout:
        for r in results:
            fout.write(json.dumps(r, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    main("LRS_VQA/image", "LRS_VQA/LRS_VQA_merged.jsonl", "/home/zhangjunjie/CaoXiongyong_Student/fubowen/LRS_VQA/dataset/LRS_VQA_RL_V2.jsonl")
