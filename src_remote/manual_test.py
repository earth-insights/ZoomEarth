from transformers import AutoProcessor, Qwen2VLForConditionalGeneration, Qwen2_5_VLForConditionalGeneration
from PIL import Image
import torch

def resize_image(image, max_size=1024):
    w, h = image.size
    scale = max_size / max(w, h)
    if scale < 1:
        new_w = int(w * scale)
        new_h = int(h * scale)
        image = image.resize((new_w, new_h), Image.BICUBIC)
    return image

# 设置环境以处理大图像
Image.MAX_IMAGE_PIXELS = None

# 加载模型和处理器
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-3B-Instruct",
    device_map="auto",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
).eval()

processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct", trust_remote_code=True)

def ask_question():
    # 获取用户输入的问题和图像路径
    question = input("请输入问题：")
    image_path = input("请输入图像路径：")

    # 准备消息
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                # {"type": "text", "text": question + "\nChoose the most appropriate answer from the above four options and just reply with one of the letters A, B, C or D."},
                {"type": "text", "text": question},
            ],
        },
    ]

    # 打开并处理图像
    try:
        image = Image.open('/home/zhangjunjie/CaoXiongyong_Student/fubowen/LRS_VQA/LRS_VQA/image/'+image_path).convert("RGB")
        image = resize_image(image)
    except Exception as e:
        print(f"打开图像时出错：{e}")
        return

    # 处理输入的文本和图像
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors="pt").to(model.device)

    # 生成回答
    generate_ids = model.generate(**inputs, 
                                  max_new_tokens=128,
                                  do_sample=True,
                                  temperature=0.7,
                                  top_k=50)
    outputs = processor.tokenizer.batch_decode(generate_ids, skip_special_tokens=True)[0].strip()

    # 打印回答
    print("回答：", outputs.split('assistant\n')[-1])

# 无限循环，只要有输入就执行
while True:
    ask_question()
