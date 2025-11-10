import os
import json
from PIL import Image
import torch
from tqdm import tqdm
import shortuuid
from functools import partial
from torch.utils.data import DataLoader
from accelerate import Accelerator
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLProcessor
from datasets import load_from_disk
import re
import os
import copy
import argparse

BATCH_SIZE = 1
Image.MAX_IMAGE_PIXELS = None 

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
        # 中心点
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2

        # 初步计算边界
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

def collate_fn(examples):
    return examples

def prepare_dataloader(ds_path, collate_fn):
    mixed_dataset_val = load_from_disk(ds_path)
    val_dataloader = DataLoader(
        mixed_dataset_val,
        batch_size=BATCH_SIZE,
        collate_fn=collate_fn,
        shuffle=False,
        num_workers=1,
    )
    return val_dataloader

def chat_batch(prompts, imgs, processor, accelerator, model):
    inputs = processor(
        text=prompts,
        images=imgs,
        return_tensors="pt",
        padding="longest"
    ).to(accelerator.device)

    gen_ids = accelerator.unwrap_model(model).generate(
        **inputs,
        max_new_tokens=1024,
        do_sample=True,
        num_beams=1,
        temperature=0.01,
    )
    
    outputs = [] 
    input_lens = inputs["input_ids"].shape[1] 
    gen_ids = gen_ids[:, input_lens:] 
    for i in range(len(gen_ids)): 
        outputs.append( 
            processor.tokenizer.decode(gen_ids[i], skip_special_tokens=True).strip() 
            ) 
    return outputs

def record(fout, prompt, sample, data, output1, output2, is_error):
    fout.write(json.dumps({
        "question_id":    sample["question_id"],
        "ground_truth":   sample["ground_truth"],
        "answer1":        extract_answer(output1),
        "answer2":        extract_answer(output2),
        "bbox_ref":       data["bbox"],
        "bbox":           extract_bbox(output1, 1),
        "prompt":         prompt,
        "category":       sample["category"],
        "stage1":         output1,
        "stage2":         output2,
        "type":           sample["type"],
        "image":          sample["image_name"],
        "error":          is_error,
        "model_id":       "ZoomEarth---LRS-GRO"
    }, ensure_ascii=False) + "\n")
    fout.flush()

def eval_model_lora(model_name, exp_name):

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
    )
    model.eval()

    processor = Qwen2_5_VLProcessor.from_pretrained(
        model_name, trust_remote_code=True,
        max_pixels = 128*128*28*28
    )
    processor.tokenizer.padding_side = "left"
    accelerator = Accelerator(mixed_precision="bf16", project_dir="checkpoints", log_with=[])

    model.generation_config.temperature = 0.01
    model.generation_config.top_p = None
    model.generation_config.top_k = None

    os.makedirs("results", exist_ok=True)
    rank = accelerator.process_index
    out_path = f"results/{exp_name}{rank}.jsonl"
    fout = open(out_path, "w", encoding="utf-8")

    test_collate_fn = partial(collate_fn)
    dataloader = prepare_dataloader("./LRS_GRO/test", collate_fn=test_collate_fn)
    model, dl = accelerator.prepare(model, dataloader)
    # cnt = 0
    for examples in tqdm(dl, desc="Evaluating"):
            tag = False
            texts = []
            images = []
            scales = []
            cur_prompts = []
            image_fps = []
            prefix = """
<|im_start|>system
You are a helpful assistant. <|im_end|>
<|im_start|>user
<|vision_start|><|image_pad|><|vision_end|>"""
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
<|im_end|><|im_start|>assistant
"""
            for example in examples:
                sample = example
                cur_prompt = sample["question"]
                image_fp = "./image/"+sample["image_name"].split("/")[-1]
                text = prefix + cur_prompt + instruction
                texts.append(text)
                image, scale = resize_image(Image.open(image_fp).convert("RGB"))
                images.append(image)
                scales.append(scale)
                cur_prompts.append(cur_prompt)
                image_fps.append(image_fp)
            if tag:
                continue
            prompts_ori = copy.deepcopy(texts)
            outputs1 = chat_batch(texts, images, processor, accelerator, model)
            # stage 2: Question + Image (downsampled) + previous reasoning + Image (cropped)
            prompts_2 =  [prompts_ori[i] + outputs1[i].split("<answer>")[0] + "<|vision_start|><|image_pad|><|vision_end|>" for i in range(len(texts))]
            bboxs = [extract_bbox(outputs1[i], scales[i]) for i in range(len(outputs1))]
            stage_2_prompts = []
            stage_2_images = []
            stage_2_examples = []
            stage_2_outputs1 = []
            stage_2_samples = []
            stage_2_prompts_ori = []
            for i in range(len(bboxs)):
                if not bboxs[i] or bboxs[i] == []:
                    record(fout, cur_prompts[i], examples[i], examples[i], outputs1[i], "", True)
                else:
                    image_bbox = Image.open(image_fps[i]).convert("RGB")
                    cur_bbox = bboxs[i][0]
                    image_bbox, _scale = resize_image(cut_image(image_bbox, cur_bbox))
                    stage_2_prompts.append(prompts_2[i])
                    stage_2_prompts_ori.append(cur_prompts[i])
                    stage_2_images.append([images[i], image_bbox])
                    stage_2_examples.append(examples[i])
                    stage_2_outputs1.append(outputs1[i])
                    stage_2_samples.append(examples[i])
            if len(stage_2_prompts):
                outputs2 = chat_batch(stage_2_prompts, stage_2_images, processor, accelerator, model)
                for i in range(len(outputs2)):
                    record(fout, stage_2_prompts_ori[i], stage_2_samples[i], stage_2_examples[i], stage_2_outputs1[i], outputs2[i], False)
    if accelerator.is_main_process:
        fout.close()
        print("Done! Predictions has been written to: ", out_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate LoRA model")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the model")
    parser.add_argument("--exp_name", type=str, required=True, help="Experiment name")
    args = parser.parse_args()
    eval_model_lora(args.model_name, args.exp_name)