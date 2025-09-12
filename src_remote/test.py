from transformers import Qwen2VLProcessor
# from model import Qwen2VLForConditionalGeneration
from datasets import load_from_disk
from torch.utils.data import DataLoader
from PIL import Image
import torch
Image.MAX_IMAGE_PIXELS = None

if not hasattr(torch, "compiler"):
    import types
    torch.compiler = types.SimpleNamespace()

if not hasattr(torch.compiler, "is_compiling"):
    torch.compiler.is_compiling = lambda: False

def collate_fn(examples):
    return examples

def prepare_dataloader(ds_path, collate_fn):
    mixed_dataset_val = load_from_disk(ds_path)
    val_dataloader = DataLoader(
        mixed_dataset_val,
        batch_size=2,
        collate_fn=collate_fn,
        shuffle=False,
        num_workers=1,
    )
    return val_dataloader

def resize_image(image, max_size=1024):
    w, h = image.size
    scale = max_size / max(w, h)
    if scale < 1:
        new_w = int(w * scale)
        new_h = int(h * scale)
        image = image.resize((new_w, new_h), Image.BICUBIC)
    return image

def main():
    # model_name: str = "/home/zhangjunjie/.cache/huggingface/hub/models--Qwen--Qwen2-VL-7B-Instruct/snapshots/eed13092ef92e448dd6875b2a00151bd3f7db0ac"
    # base_model = Qwen2VLForConditionalGeneration.from_pretrained(
    #     model_name, torch_dtype=torch.float16, trust_remote_code=True
    # )
    processor = Qwen2VLProcessor.from_pretrained(
        "Qwen/Qwen2-VL-7B-Instruct", trust_remote_code=True, max_pixels=128*128*28*28
    )

    # print("151655 ->", processor.convert_ids_to_tokens(151655))  # <|image_pad|>
    # print("151643 ->", processor.convert_ids_to_tokens(151643))  # <|image_0|>
    # print("151644 ->", processor.convert_ids_to_tokens(151644))  # <|image_1|>

    dl = prepare_dataloader("./LRS_VQA_RL/rl", collate_fn)

    for example in dl:
        image_fp0 = "./LRS_VQA/image/"+example[0]["image_name"].split("/")[-1]
        image_fp1 = "./LRS_VQA/image/"+example[1]["image_name"].split("/")[-1]
        img0 = Image.open(image_fp0).convert("RGB")
        img0 = resize_image(img0)
        img1 = Image.open(image_fp1).convert("RGB")
        img1 = resize_image(img1)
        prompt0 = "<|image_pad|>\n" + example[0]["question"]
        prompt1 = "<|image_pad|>\n" + example[1]["question"]
        inputs = processor(
            text=[prompt0, prompt1],
            images=[[img0],[img1]],
            return_tensors="pt",
            padding="longest"
        ).to("cuda")
        print("hi")
if __name__ == "__main__":
    main()

