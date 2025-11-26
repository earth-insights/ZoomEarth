from transformers import Qwen2_5_VLProcessor, AutoModelForCausalLM
from PIL import Image
import re
import torch

def chat_batch(prompts, imgs, processor, model):
    inputs = processor(
        text=prompts,
        images=imgs,
        return_tensors="pt",
        padding="longest"
    ).to(model.device)

    gen_ids = model.generate(
        **inputs,
        max_new_tokens=1024,
        do_sample=False,
        num_beams=1
    )
    
    outputs = [] 
    input_lens = inputs["input_ids"].shape[1] 
    gen_ids = gen_ids[:, input_lens:] 
    for i in range(len(gen_ids)): 
        outputs.append( 
            processor.tokenizer.decode(gen_ids[i], skip_special_tokens=True).strip() 
            ) 
    return outputs

def cut_image(image, bbox, min_size=512):
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
        return cropped

def extract_bbox(completion_content: str, scale):
    pattern = r'"bbox_2d"\s*:\s*\[(.*?)\]'
    matches = re.findall(pattern, completion_content, re.DOTALL)

    bboxes = []
    for m in matches:
        try:
            nums = [int(x.strip()) for x in m.split(",")]
            bbox = [num * scale for num in nums]
            bboxes.append(bbox)
        except ValueError:
            continue
    return bboxes

def resize_image(image, max_size=1024):
    w, h = image.size
    scale = max_size / max(w, h)
    if scale < 1:
        new_w = int(w * scale)
        new_h = int(h * scale)
        image = image.resize((new_w, new_h), Image.BICUBIC)
    return image

MODEL_PATH = ""
PREFIX = """
<|im_start|>system
You are a helpful assistant. <|im_end|>
<|im_start|>user
<|vision_start|><|image_pad|><|vision_end|>"""
INSTRUCTION = """
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
<|im_end|><|im_start|>assistant
"""

def chat(prompt, image_fp):
    processor = Qwen2_5_VLProcessor.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)

    prompts = [prompt + INSTRUCTION]
    image = Image.open(image_fp).convert("RGB")
    scale = max(1, max(image.width, image.height)/1024)
    images = [resize_image(image)]
    output1 = chat_batch(prompts, images, processor, model)[0]

    bboxs = extract_bbox(output1, scale)
    if bboxs !=[]:
        bbox = bboxs[0]
        image_bbox = Image.open(image_fp).convert("RGB")
        image_bbox = resize_image(cut_image(image_bbox, bbox))
        images.append(image_bbox)
        new_prompt = prompt + INSTRUCTION + output1.split("<answer>")[0] + "<|vision_start|><|image_pad|><|vision_end|>"
        output2 = chat_batch([new_prompt], images, processor, model)[0]
        return output2
    else:
        return output1

if __name__ == "__main__":
    prompt = "Are there any building on the top-right island?"
    image_fp = "./images/demo3.png"
    output = chat(prompt=prompt, image_fp=image_fp)
    print(output)
    
    
