import os
import json
import base64
from tqdm import tqdm
from openai import OpenAI
from PIL import Image
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor, as_completed

# 使用 OpenAI SDK 兼容 DashScope API
client = OpenAI(
    api_key = "sk-17f05c023b1a4122909ace0c77d69a49",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
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
    return f"data:{mime_type};base64,{encoded}"

def process_item(item, image_dir, model_name="qwen2-vl-72b-instruct"):
    result = {
        "stage_1_reasoning": "",
        "stage_2_reasoning": "",
        "image_name": item.get("image").split("/")[-1],
        "question_id": item.get("question_id"),
        "question": item.get("text"),
        "ground_truth": item.get("ground_truth"),
        "cut": False,
        "category": item.get("category"),
        "global": "",
        "area": "",
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
Based on the given image, question and ground truth, generate your response
- Two images are provided: the first is the original downsampled satellite image; the second shows the area where the true answer can be found.
- First generate a description of the global picture
- Then, I need reasoning for two stages.
- The first stage is about the global picture; you need to reason about how to crop the second image from the first based on the question.
- The second stage is about the cropped image; you need to reason about how to find the answer in the cropped image.
- Response as below:
<global>
- What is this RS image about. (e.g. there's a river at the left part of the image.)
<\\global>
<stage_1_reasoning>
- Question Intent: Describe the question (e.g., object category, color, quantity, etc.).
- Object Localization: Infer where the reference object might appear in the image. Please mention any ambiguity or lack of clarity in the spatial reference.
- Reasoning Strategy: Describe how to locate and analyze the relevant regions in the image to answer the question.
- Reasoning Result: Where should the crop be made?
<\\stage_1_reasoning>
<stage_2_reasoning>
- Describe what you should do after narrowing down your searching area, that is, what you should do in the second inmage.
<\\stage_2_reasoning>

Input:
- Question: {result["question"]}
- Ground Truth: {result["ground_truth"]}

"""

        messages = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "You are an expert in visual reasoning over satellite images. "
                            "Given one or more images and a question, your task is to explain your reasoning "
                            "on how to locate the target and answer the question. "
                            "You must not provide the answer itself — only your reasoning steps."
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
            model=model_name,
            messages=messages,
            temperature=0.6
        )
        answer = response.choices[0].message.content

        result.update({
            "stage_1_reasoning": answer.split("<stage_1_reasoning>")[-1].split("<\\stage_1_reasoning>")[0],
            # "stage_1_reasoning": answer,
            "stage_2_reasoning": answer.split("<stage_2_reasoning>")[-1].split("<\\stage_2_reasoning>")[0],
            "cut": True,
            "global": answer.split("<global>")[-1].split("<\\global>")[0],
            "area": area,
            "bbox": [int(x / scale) for x in hbox],
            "scale": scale
        })
    except Exception as e:
        result["cut"] = True
        result["stage_1_reasoning"] = f"Error: {str(e)}"

    return result

def main(image_dir, jsonl_in, jsonl_out, model_name="qwen2-vl-72b-instruct", max_workers=50):
    with open(jsonl_in, 'r', encoding='utf-8') as fin:
        items = [json.loads(line) for line in fin][:1]

    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(process_item, item, image_dir, model_name)
            for item in items
        ]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
            results.append(future.result())

    with open(jsonl_out, 'w', encoding='utf-8') as fout:
        for r in results:
            fout.write(json.dumps(r, ensure_ascii=False) + "\n")

# def main(image_dir, jsonl_in, jsonl_out, model_name="qwen2-vl-72b-instruct"):
#     results = []

#     with open(jsonl_in, 'r', encoding='utf-8') as fin:
#         for line in tqdm(fin, desc="Processing"):
#             item = json.loads(line)
#             imgname = item.get("image").split("/")[-1]
#             question_text = item.get("text")
#             ground_truth = item.get("ground_truth")

#             image_path = os.path.join(image_dir, imgname)
#             if not os.path.exists(image_path):
#                 continue

#             prompt = f"""
# <reasoning>
# - Question Intent: Describe what the question is asking (e.g., object category, color, count, etc.).
# - Target Localization: Infer where the referenced object(s) are likely to appear in the image. Mention any ambiguity or vagueness in spatial references.
# - Reasoning Strategy: Describe how you would locate and analyze the relevant region in the image to answer the question.
# <\\reasoning>
# <summary>
# - Describe what you should do after narrowing down your searching area, that is, what you should do in the second inmage.
# <\\summary>

# Example:
# <reasoning>
# The question is asking about the status of the largest dock.
# To answer it, I need to find the dock region in the image.
# I can locate it by identifying all docks and selecting the one with the largest area.
# <\\reasoning>
# <summary>
# I should compare all the docks and find the largest one. Then i should check its status.
# <\\summary>

# Input:
# - Question: {question_text}
# - Ground Truth: {ground_truth}

# Note:
# - Two images are provided: the first is a downsampled original satellite image; the second shows the region where the ground truth answer can be found.
# - You **must only refer to the first image** when reasoning.
# - Do **not** mention or reference the second image in any way.
# - Avoid using phrases like "the first image" or "the second image" — write as if only the main image was provided.
# - Only output your **reasoning process**, do **not** provide the final answer to the question.
# - The summary sentence **do not** contain the answer, only describing what you should do to get the final answer.
# """
            
#             if not item.get("hbox", None):
#                 results.append({
#                     "image_name": imgname,
#                     "question_id": item.get("question_id"),
#                     "question": question_text,
#                     "ground_truth": ground_truth,
#                     "cut": False,
#                     "category": item.get("category"),
#                     "stage_1_reasoning": "",
#                     "stage_2_reasoning": "",
#                     "area": "",
#                     "bbox": [],
#                     "scale": 1
#                 })
#                 continue

#             try:
#                 image1 = load_image_from_path(image_path)
#                 w, h = image1.size
#                 scale = max(max(w, h) / 1024, 1)

#                 center_x = (item.get("hbox")[0] + item.get("hbox")[2]) / 2
#                 center_y = (item.get("hbox")[1] + item.get("hbox")[3]) / 2

#                 area = ""

#                 if center_y < h / 3:
#                     area = "top-"
#                 elif center_y > 2 * h / 3:
#                     area = "bottom-"
#                 else:
#                     area = "center-"

#                 if center_x < w / 3:
#                     area += "left"
#                 elif center_x > 2 * w / 3:
#                     area += "right"
#                 else:
#                     if area == "center-":
#                         area = "center"
#                     else:
#                         area += "center"
                

#                 image2 = cut_image(image1, item.get("hbox"))
#                 image1 = resize_image(image1)

#                 image_url_1 = encode_pil_image_to_data_url(image=image1)
#                 image_url_2 = encode_pil_image_to_data_url(image=image2)

#                 messages = [
#                     {
#                         "role": "system",
#                         "content": [
#                             {
#                                 "type": "text",
#                                 "text": (
#                                     "You are an expert in visual reasoning over satellite images. "
#                                     "Given one or more images and a question, your task is to explain your reasoning "
#                                     "on how to locate the target and answer the question. "
#                                     "You must not provide the answer itself — only your reasoning steps."
#                                 )
#                             }
#                         ]
#                     },
#                     {
#                         "role": "user",
#                         "content": [
#                             {"type": "image_url", "image_url": {"url": image_url_1}},
#                             {"type": "image_url", "image_url": {"url": image_url_2}},
#                             {"type": "text", "text": prompt}
#                         ]
#                     }
#                 ]

#                 response = client.chat.completions.create(
#                     model=model_name,
#                     messages=messages
#                 )
#                 answer = response.choices[0].message.content
#             except Exception as e:
#                 answer = f"Error: {str(e)}"

#             results.append({
#                 "image_name": imgname,
#                 "question_id": item.get("question_id"),
#                 "question": question_text,
#                 "ground_truth": ground_truth,
#                 "cut": True,
#                 "category": item.get("category"),
#                 "stage_1_reasoning": answer.split("<reasoning>")[-1].split("<\\reasoning>")[0],
#                 "stage_2_reasoning": answer.split("<summary>")[-1].split("<\\summary>")[0],
#                 "area": area,
#                 "bbox": [int(x / scale) for x in item.get("hbox")],
#                 "scale": scale
#             })

#             break

#     with open(jsonl_out, 'w', encoding='utf-8') as fout:
#         for r in results:
#             fout.write(json.dumps(r, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    main("LRS_VQA/image", "LRS_VQA/LRS_VQA_merged.jsonl", "dataset/vl_answers.jsonl")
