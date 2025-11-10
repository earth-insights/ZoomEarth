import os
import json
import torch
import argparse
from PIL import Image
from tqdm import tqdm
from functools import partial
from dataclasses import dataclass
from datetime import datetime, date

from model import Qwen2VLProcessor
from accelerate import Accelerator
from datasets import load_from_disk
from accelerate.utils import set_seed
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from transformers import Qwen2_5_VLForConditionalGeneration

Image.MAX_IMAGE_PIXELS = None

@dataclass
class TrainingConfig:
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-VL-3B-Instruct",
        dataset_name_train: str = "DATASET_PTH/sft",
        dataset_name_val: str = "DATASET_PTH/test",
        output_dir: str = "OUT_DIR",
        img_folder: str = "IMAGE_FOLDER",
        wandb_project: str = None,
        num_train_epochs: int = 3,
        batch_size_per_gpu: int = 1,
        gradient_accumulation_steps: int = 4,
        lr: float = 3e-5,
        lora_r: int = 8,
        warmup_steps: int = 500,
        weight_decay: float = 0.01,
        max_grad_norm: float = 1.0,
        seed: int = 42,
        dtype = torch.bfloat16,
        quantization: bool = False,
        resume_from_checkpoint: bool = True,
        save_steps: int = 100,
        log_steps: int = 10,
        print_steps: int = 20,
        max_pixels: int = 64*64*28*28,
    ):
        self.model_name = model_name
        self.dataset_name_train = dataset_name_train
        self.dataset_name_val = dataset_name_val
        self.output_dir = output_dir
        self.img_folder = img_folder

        if wandb_project is None:
            self.wandb_project = f"QwenVL-LRS-GRO-{date.today()}_{datetime.now().time()}".replace(":", "_")
        else:
            self.wandb_project = wandb_project

        self.num_train_epochs = num_train_epochs
        self.batch_size_per_gpu = batch_size_per_gpu
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.lr = lr
        self.lora_r = lora_r
        self.warmup_steps = warmup_steps
        self.weight_decay = weight_decay
        self.max_grad_norm = max_grad_norm
        self.seed = seed
        self.dtype = dtype
        self.quantization = quantization
        self.resume_from_checkpoint = resume_from_checkpoint
        self.save_steps = save_steps
        self.log_steps = log_steps
        self.print_steps = print_steps
        self.max_pixels = max_pixels

def resize_image(image, max_size=1024):
    w, h = image.size
    scale = max_size / max(w, h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    return image.resize((new_w, new_h), Image.BICUBIC)

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
        w, h = cropped.size
        scale = min_size / min(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cropped.resize((new_w, new_h), Image.BICUBIC)

        left = (new_w - min_size) // 2
        top = (new_h - min_size) // 2
        return resized.crop((left, top, left + min_size, top + min_size))

def collate_fn(examples, processor, dtype, img_folder):

    texts = []
    labels = []
    images = []

    for example in examples:
        text = "<|image_pad|> \n" + example["question"] + """
Task:
1. Global view – Give a one-sentence description of the entire scene.
2. Reasoning focus – Decide which part of the image you must attend to in order to answer the question. Wrap the chosen keyword (pick exactly one from bottom-left, bottom-right, bottom-center, top-left, top-right, top-center, center-left, center-right, center) in the tag <location>...</location>.
3. Answer box – Output the bounding box of that region as pixel coordinates in the form <bbox>[x1,y1,x2,y2]</bbox>. Use integers, no spaces.
4. Post-crop analysis - After cropping to the box in step 3, examine that patch and write a brief statement explaining the visual evidence that supports your answer.
5. Answer - your answer. In the tag <answer>...</answer>

Rules:
- Return exactly one <location> tag and one <bbox> tag; nothing else after them.
- If unsure, pick the most probable location and best-guess box—never say you are uncertain.
"""
        texts.append(text)

        if example['cut']:
            label = (
                f"{example['global']}"
                f" {example['stage_1_reasoning']} "
                f"<location>{example['area']}</location>\n"
                f"<bbox>{example['bbox']}</bbox>\n"
                f"<|image_pad|>\n"
                f"{example['stage_2_reasoning']}\n"
                f"<answer>{example['ground_truth']}</answer> <|endoftext|>"
            )
            image1 = Image.open(img_folder + '/'+example["image_name"]).convert("RGB")
            image2 = cut_image(image=image1, bbox=example['bbox'])
            image1 = resize_image(image=image1)
            images.append([image1, image2])
        else:
            label = (
                f"{example['global']}"
                f" {example['stage_1_reasoning']} "
                f"{example['stage_2_reasoning']}\n"
                f"<answer>{example['ground_truth']}</answer> <|endoftext|>"
            )
            images.append(resize_image(Image.open(img_folder + '/'+example["image_name"]).convert("RGB")))
        labels.append(label)

    tokens = processor(text=texts, images=images, return_tensors="pt", padding="longest", text_pair=labels)
    return tokens.to(dtype)

def prepare_model_and_processor(config: TrainingConfig):
    processor = Qwen2VLProcessor.from_pretrained(config.model_name, max_pixels = config.max_pixels)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(config.model_name, torch_dtype="auto")
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

def parse_args():
    parser = argparse.ArgumentParser(description="Training Configuration")

    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--dataset_name_train", type=str, default="DATASET_PTH/sft")
    parser.add_argument("--dataset_name_val", type=str, default="DATASET_PTH/test")
    parser.add_argument("--output_dir", type=str, default="OUT_DIR")
    parser.add_argument("--img_folder", type=str, default="IMAGE_FOLDER")
    parser.add_argument("--wandb_project", type=str, default=None)

    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--batch_size_per_gpu", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dtype", type=str, choices=["bfloat16", "float16"], default="bfloat16")
    parser.add_argument("--quantization", action="store_true")
    parser.add_argument("--resume_from_checkpoint", action="store_true")

    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--log_steps", type=int, default=10)
    parser.add_argument("--print_steps", type=int, default=20)
    parser.add_argument("--max_pixels", type=int, default=64*64*28*28)

    return parser.parse_args()

def train(args):
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    config = TrainingConfig(
        model_name=args.model_name,
        dataset_name_train=args.dataset_name_train,
        dataset_name_val=args.dataset_name_val,
        output_dir=args.output_dir,
        img_folder=args.img_folder,
        num_train_epochs=args.num_train_epochs,
        batch_size_per_gpu=args.batch_size_per_gpu,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        lr=args.lr,
        lora_r=args.lora_r,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        seed=args.seed,
        dtype=dtype,
        quantization=args.quantization,
        resume_from_checkpoint=args.resume_from_checkpoint,
        save_steps=args.save_steps,
        log_steps=args.log_steps,
        print_steps=args.print_steps,
        max_pixels=args.max_pixels,
    )

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

    dataloader, val_dataloader, model, optimizer, scheduler = accelerator.prepare(
        dataloader, val_dataloader, model, optimizer, lr_scheduler
    )

    progress_bar = tqdm(total=num_training_steps, disable=not accelerator.is_local_main_process)
    starting_epoch = 1
    global_step = 0
    
    skipped_dataloader = dataloader
    if config.resume_from_checkpoint and os.path.exists(f"{config.output_dir}/training_info.json"):
        with open(f"{config.output_dir}/training_info.json", "r") as f:
            training_info = json.load(f)
        
        starting_epoch = training_info["epoch"]
        global_step = training_info["step"]
        load_checkpoint(accelerator, training_info["latest_checkpoint"], model)
        
        progress_bar.update(global_step)
        accelerator.print(f"Resumed from checkpoint: {training_info['latest_checkpoint']}")
        
        skip_batch_count = global_step * config.gradient_accumulation_steps % len(dataloader)
        skipped_dataloader = accelerator.skip_first_batches(dataloader, num_batches=skip_batch_count)
    
    accelerator.print(f"Starting epoch: {starting_epoch}, Global step: {global_step}")
    
    total_loss = torch.tensor(0.0, device=accelerator.device)
    total_loss_count = 0
    dataloader_step = 0
    grad_norm = None

    accelerator.init_trackers(
        project_name=config.wandb_project,
        init_kwargs={
            "wandb": {"name": f"{datetime.now().strftime('%Y-%m-%d %H-%M-%S')}"},
            "tensorboard": {}
        },
        config=vars(config))
    accelerator.print(f"Trackers: {accelerator.trackers}")
    
    unwrapped = accelerator.unwrap_model(model)
    total_params = sum(p.numel() for p in unwrapped.parameters())
    trainable_params = sum(p.numel() for p in unwrapped.parameters() if p.requires_grad)
    print(f"🔥 Trainable params: {trainable_params} / {total_params} ({trainable_params/total_params*100:.4f}%)")

    for epoch in range(1, config.num_train_epochs+1):
        if epoch < starting_epoch:
            continue
        train_dataloader = skipped_dataloader if epoch == starting_epoch else dataloader
        model.train()
        for batch in train_dataloader:
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
    args = parse_args()
    train(args)