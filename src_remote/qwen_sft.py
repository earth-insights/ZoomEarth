## Usage: run `accelerate config` in shell to config accelerate first, 
## then `accelerate launch --num_processes=8 src/qwen_sft.py` to train

import os
os.environ["ACCELERATE_LOGGING_DIR_VERSION"] = "1"
import json
import torch
from dataclasses import dataclass
from datetime import datetime
from tqdm import tqdm
from functools import partial
from torch.utils.data import DataLoader
from accelerate import Accelerator, PartialState
from accelerate.utils import set_seed
from transformers import (
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig,
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup
)
from model import Qwen2VLProcessor, Qwen2VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
from peft import LoraConfig, get_peft_model
from datasets import load_from_disk
from evaluate import load
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
import numpy as np
from datetime import datetime
from datetime import date
from copy import deepcopy
from peft import PeftModel

# os.environ["CUDA_VISIBLE_DEVICES"] = "2,6"  # 指定单个 GPU

@dataclass
class TrainingConfig:
    # model_name: str = "/root/.cache/huggingface/hub/models--Qwen--Qwen2-VL-7B-Instruct/snapshots/eed13092ef92e448dd6875b2a00151bd3f7db0ac"
    model_name: str = "/home/zhangjunjie/.cache/huggingface/hub/models--Qwen--Qwen2-VL-7B-Instruct/snapshots/eed13092ef92e448dd6875b2a00151bd3f7db0ac"
    dataset_name_train: str = "./LRS_VQA_RL/sft"
    dataset_name_val: str = "./LRS_VQA_RL/test"
    output_dir: str = "output"
    img_folder = "./LRS_VQA/image"
    
    wandb_project: str = f"QwenVL-LRS-VQA-RL-{date.today()}{datetime.now().time()}".replace(":", "_")

    num_train_epochs: int = 5
    batch_size_per_gpu: int = 1
    gradient_accumulation_steps: int = 4
    lr: float = 3e-5 #3e-5
    lora_r: int = 8
    warmup_steps: int = 500
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0

    seed: int = 42
    dtype = torch.bfloat16
    quantization: bool = False

    resume_from_checkpoint: bool = False
    save_lora_adapter_when_checkpointing: bool = True

    save_steps: int = 200
    log_steps: int = 20 # log to wandb & tensorboard, gathered loss (slower)
    print_steps: int = 40 # local print, loss on GPU0

    max_pixels = 64*64*28*28 #  token数量最大限制64*64

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM"
    )

def resize_image(image, max_size=1024):
    """缩放图像到最大边为max_size，保持比例"""
    w, h = image.size
    scale = max_size / max(w, h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    return image.resize((new_w, new_h), Image.BICUBIC)

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

# def collate_fn(examples, processor, dtype):

#     texts =[]
#     labels = [example["output"] for example in examples]
#     images = []
#     for example in examples:
#         text = "<|image_pad|>" + "<|image_pad|>" + "<|image_pad|>" + "<|image_pad|>" + "<|image_pad|>" + "<|image_pad|>" + "answer " + example["input"]  # 构造字符串  6张图
#         texts.append(text)  # 将结果添加到列表中
#         camera_views = [
#             example[cam]
#             for cam in [
#                 "cam_front",
#                 "cam_front_right",
#                 "cam_front_left",
#                 "cam_back",
#                 "cam_back_left",
#                 "cam_back_right",
#             ]
#         ]
#         images.append([Image.open(cam.replace("/root", "/cephfs")).convert("RGB") for cam in camera_views])
        
#     tokens = processor(text=texts, images=images, 
#                         return_tensors="pt", padding="longest", suffix=labels)
#     return tokens.to(dtype), f"{example['scene']}_{example['sample_token']}"

def collate_fn(examples, processor, dtype, img_folder):

    texts = []
    labels = []
    images = []

    for example in examples:
        text = "<|image_pad|> \n" + example["question"]  # <image> how are you ?  # <|placeholder|> <|placeholder|> ...<|placeholder|>  how are you ?
        texts.append(text)

        # 构建输出：包含多个 reasoning 阶段的结构化文本
        if example['cut']:
            label = (
                f"<global>{example['global']}</global>\n"
                f"<stage_1_reasoning>{example['stage_1_reasoning']}</stage_1_reasoning>\n"
                f"<location>{example['area']}</location>\n"
                f"<bbox>{example['bbox']}</bbox>\n"
                f"<|image_pad|>\n"
                f"<stage_2_reasoning>{example['stage_2_reasoning']}</stage_2_reasoning>\n"
                f"<ground_truth>{example['ground_truth']}</ground_truth> <|endoftext|>"
            )
            # 用两张全图图像
            image1 = Image.open(img_folder + '/'+example["image_name"]).convert("RGB")
            image2 = cut_image(image=image1, bbox=example['bbox'])
            image1 = resize_image(image=image1)
            images.append([image1, image2])
        else:
            label = (
                f"<ground_truth>{example['ground_truth']}</ground_truth> <|endoftext|>"
            )
            # 只用一张全图图像
            images.append(resize_image(Image.open(img_folder + '/'+example["image_name"]).convert("RGB")))
        labels.append(label)


        

    tokens = processor(text=texts, images=images, return_tensors="pt", padding="longest", text_pair=labels)
    # print('hello')
    return tokens.to(dtype), None  # 你可以用 id 也可以不返回


def prepare_model_and_processor(config: TrainingConfig):
    processor = Qwen2VLProcessor.from_pretrained(config.model_name, max_pixels = config.max_pixels)
    if config.quantization:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=config.dtype
        )
        model = Qwen2VLForConditionalGeneration.from_pretrained(
    config.model_name, torch_dtype="auto")
    else:
        model = Qwen2VLForConditionalGeneration.from_pretrained(
    config.model_name, torch_dtype="auto")
    # for param in model.parameters():
    #     param.requires_grad = False
    # for param in model.lm_head.parameters():
    #     param.requires_grad = True
    model = get_peft_model(model, config.lora_config)

    return model, processor

def prepare_dataloader(config: TrainingConfig, collate_fn):
    mixed_dataset = load_from_disk(config.dataset_name_train)
    train_dataloader = DataLoader(
        mixed_dataset,
        batch_size=config.batch_size_per_gpu,
        collate_fn=collate_fn,
        shuffle=False,
        num_workers=1,
    )
    mixed_dataset_val = load_from_disk(config.dataset_name_val)
    val_dataloader = DataLoader(
        mixed_dataset_val,
        batch_size=config.batch_size_per_gpu,
        collate_fn=collate_fn,
        shuffle=False,
        num_workers=1,
    )
    return train_dataloader, val_dataloader

def prepare_optimizer_and_scheduler(config: TrainingConfig, model, num_training_steps):
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.warmup_steps,
        num_training_steps=num_training_steps,
    )
    return optimizer, lr_scheduler

def save_checkpoint(accelerator, model, epoch, step, config, loss, checkpoint_dir=None):
    if checkpoint_dir is None:
        checkpoint_dir = f"{config.output_dir}/checkpoint-{step}"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    if config.save_lora_adapter_when_checkpointing:
        save_lora_adapter(accelerator, model, checkpoint_dir)
    
    training_info = {
        "epoch": epoch,
        "step": step,
        "loss": loss,
        "latest_checkpoint": checkpoint_dir,
    }
    with open(f"{config.output_dir}/training_info.json", "w") as f:
        json.dump(training_info, f)
    
    accelerator.save_state(checkpoint_dir, safe_serialization=False)

def load_checkpoint(accelerator, checkpoint_dir):
    accelerator.print("loading checkpoint")
    accelerator.load_state(checkpoint_dir)

def save_lora_adapter(accelerator, model, path):
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.save_pretrained(
        path,
        is_main_process=accelerator.is_main_process,
        save_function=accelerator.save,
    )
    

def train():
    config = TrainingConfig()

    set_seed(config.seed)

    accelerator = Accelerator(
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        mixed_precision='bf16',
        log_with=["tensorboard"],
        project_dir=config.output_dir,
    )

    model, processor = prepare_model_and_processor(config)
    train_collate_fn = partial(collate_fn, processor=processor, dtype=config.dtype, img_folder=config.img_folder)
    dataloader, val_dataloader = prepare_dataloader(config, train_collate_fn)
    
    num_training_steps = len(dataloader) * config.num_train_epochs // config.gradient_accumulation_steps
    optimizer, lr_scheduler = prepare_optimizer_and_scheduler(config, model, num_training_steps)
    
    num_training_steps = num_training_steps // accelerator.num_processes

    print("→ 准备送入 accelerator.prepare()，开始模型拷贝到设备")
    dataloader, val_dataloader, model, optimizer, scheduler = accelerator.prepare(
        dataloader, val_dataloader, model, optimizer, lr_scheduler
    )
    print("✓ accelerator.prepare() 完成")

    progress_bar = tqdm(total=num_training_steps, disable=not accelerator.is_local_main_process)
    starting_epoch = 1
    global_step = 0
    
    skipped_dataloader = dataloader
    if config.resume_from_checkpoint and os.path.exists(f"{config.output_dir}/training_info.json"):
        with open(f"{config.output_dir}/training_info.json", "r") as f:
            training_info = json.load(f)
        
        starting_epoch = training_info["epoch"]
        global_step = training_info["step"]
        load_checkpoint(accelerator, training_info["latest_checkpoint"])
        
        progress_bar.update(global_step)
        accelerator.print(f"Resumed from checkpoint: {training_info['latest_checkpoint']}")
        
        skip_batch_count = global_step * config.gradient_accumulation_steps % len(dataloader)
        skipped_dataloader = accelerator.skip_first_batches(dataloader, num_batches=skip_batch_count)
    
    accelerator.print(f"Starting epoch: {starting_epoch}, Global step: {global_step}")
    
    total_loss = torch.tensor(0.0, device=accelerator.device)
    total_loss_count = 0
    dataloader_step = 0
    grad_norm = None

    print("✅ 当前工作目录:", os.getcwd())
    print("✅ 当前输出目录:", config.output_dir)

    if accelerator.is_main_process:
        accelerator.init_trackers(
            project_name=config.wandb_project,
            init_kwargs={
                "wandb": {"name": f"{datetime.now().strftime('%Y-%m-%d %H-%M-%S')}"},
                "tensorboard": {}
            },
            config=vars(config)
        )
    accelerator.print(f"Trackers: {accelerator.trackers}")


    for epoch in range(1, config.num_train_epochs+1):
        # if epoch < starting_epoch:
        #     continue
        # train_dataloader = skipped_dataloader if epoch == starting_epoch else dataloader
        model.train()
        for batch, name in dataloader:
            # print('ok')
            with accelerator.accumulate(model):
                output = model(**batch)
                loss = output.loss

                total_loss += loss.detach()
                total_loss_count += 1
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    grad_norm = accelerator.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                
                # # debug
                # save_checkpoint(accelerator, model, epoch, global_step, config, loss.item())
                # #debug
                # break
                # debug
                # accelerator.log({"hi":1}, step=global_step)

                dataloader_step += 1
                if dataloader_step % config.gradient_accumulation_steps == 0:
                    global_step += 1
                    progress_bar.update(1)

                    if global_step % config.print_steps == 0:
                        accelerator.print(f"Epoch {epoch}, Step {global_step}, Loss {loss.item()}")

                    if global_step % config.log_steps == 0:
                        log_data = {
                            "train/loss": accelerator.gather(total_loss).detach().sum().item() / accelerator.num_processes / total_loss_count,
                            "train/learning_rate": scheduler.get_last_lr()[0],
                            "train/global_step": global_step,
                            "train/epoch": global_step / num_training_steps * config.num_train_epochs,
                            "train/grad_norm": grad_norm.detach().item() if isinstance(grad_norm, torch.Tensor) else grad_norm
                        }
                        accelerator.log(log_data, step=global_step)
                        total_loss = torch.tensor(0.0, device=accelerator.device)
                        total_loss_count = 0
                        accelerator.wait_for_everyone()
                    
                    if global_step % config.save_steps == 0:
                        save_checkpoint(accelerator, model, epoch, global_step, config, loss.item())
    save_checkpoint(accelerator, model, epoch, global_step, config, loss.item())
    accelerator.end_training()


if __name__ == "__main__":
    train()
