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

def process_item(item, image_dir, model_name="qwen2-vl-72b-instruct"):
    result = {
        "global": "",
        "stage_1_reasoning": "",
        "stage_2_reasoning": "",
        "area": "",
        "question": item.get("text"),
        "ground_truth": item.get("ground_truth"),
        "prompt": ""
    }

    image_path = os.path.join(image_dir, item.get("image").split("/")[-1])
    if not os.path.exists(image_path):
        return result

    hbox = item.get("hbox")
    if not hbox or len(hbox) != 4:
        return result
    
    try:
        image1 = load_image_from_path(image_path)
        w, h = image1.size
        scale = max(max(w, h) / 1024, 1)

        center_x = (hbox[0] + hbox[2]) / 2
        center_y = (hbox[1] + hbox[3]) / 2

        area = ""
        if center_y < h / 3:
            area = "top-"
        elif center_y > 2 * h / 3:
            area = "bottom-"
        else:
            area = "center-"
        if center_x < w / 3:
            area += "left"
        elif center_x > 2 * w / 3:
            area += "right"
        else:
            area = area if area == "center-" else area + "center"

        image2 = cut_image(image1, hbox)
        image1 = resize_image(image1)

        image_url_1 = encode_pil_image_to_data_url(image1)
        image_url_2 = encode_pil_image_to_data_url(image2)

        prompt = f"""
You are an intelligent remote sensing analyst. Your task is to reason about visual question answering using satellite imagery.

- A global satellite image is provided as input (downsampled for efficiency).
- The answer to the question lies in a specific region of this global image, which has been manually cropped out for reference.
- Your job is to:
    1. Describe the global image.
    2. Perform reasoning in two stages:
        - First: Based on the question, how should one locate the region where the object of interest lies in the global image?
        - Second: After locating this region, how should one find the final answer using the cropped region?

Structure your response strictly as follows:

<global>
- Provide a brief but informative description of the global RS image (e.g., key structures, spatial layout).
<\global>

<stage_1_reasoning>
- Question Intent: What type of question is being asked? (e.g., category, count, color, spatial relation, etc.)
- Localization Strategy: How should one locate the relevant region in the global image? What visual or textual clues guide the search?
- Reasoning Result: Where should the crop be taken from? Describe the approximate location and why.
<\\stage_1_reasoning>

<stage_2_reasoning>
- Given the cropped region, how should one analyze it to derive the final answer? Describe what should be looked for.
<\\stage_2_reasoning>

Input:
- Question: {result["question"]}
- Ground Truth Answer: {result["ground_truth"]}
- The answer-related region corresponds to the [{result["area"]}] part of the global image.
"""


        messages = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "You are an expert in visual reasoning over remote sensing (RS) imagery. "
                            "You are given a global downsampled satellite image and a question. "
                            "Your task is to explain your reasoning in two stages:\n"
                            "1. First, how to locate the region in the image where the answer is likely to be found.\n"
                            "2. Then, how to analyze that region to derive the answer.\n"
                            "You should not provide the final answer — only explain the reasoning process clearly and step-by-step."
                        )
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
            "global": answer.split("<global>")[-1].split("<\\global>")[0],
            "stage_1_reasoning": answer.split("<stage_1_reasoning>")[-1].split("<\\stage_1_reasoning>")[0],
            "stage_2_reasoning": answer.split("<stage_2_reasoning>")[-1].split("<\\stage_2_reasoning>")[0],
            "area": area,
            "prompt": prompt
        })
    except Exception as e:
        result["stage_1_reasoning"] = f"Error: {str(e)}"

    return result

def main(image_dir, jsonl_in, jsonl_out, model_name="qwen2-vl-72b-instruct", max_workers=50):
    with open(jsonl_in, 'r', encoding='utf-8') as fin:
        items = [json.loads(line) for line in fin][:10]

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
    main("LRS_VQA/image", "LRS_VQA/LRS_VQA_merged.jsonl", "dataset/prompts.jsonl")
