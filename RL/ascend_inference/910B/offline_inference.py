from transformers import AutoProcessor
from vllm import LLM, SamplingParams
from qwen_vl_utils import process_vision_info

MODEL_PATH = "VLM-R1-Qwen2.5VL-3B-OVD-0321"

llm = LLM(
    model=MODEL_PATH,
    max_model_len=16384,
    enforce_eager=True
)

sampling_params = SamplingParams(
    max_tokens=512
)

image_path = "./resources/test.jpg"
describe = "杯子在哪个位置？请输出杯子的bbox坐标。"
event = "杯子"
image_messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": f"{image_path}",
                "min_pixels": 224 * 224,
                "max_pixels": 1280 * 28 * 28,
            },
            {"type": "text", "text": f"请分析图像并回答以下问题。您的回答应包含对图像内容的简要描述和最终答案。描述使用 `<description></description>` 标签包裹，答案使用 `</think></think>` 标签包裹。答案必须以 JSON 格式输出，包含 \"yes\" 或 \"no\"，并提供相关物体的边界框坐标作为解释。如果没有涉及具体物体，则将 \"explanations\" 设为 \"None\"。输出格式如下：\n\n<description>对图像内容的简要描述写在这里</description>\n\n<|FunctionCallBegin|>\n```json\n{{\"answer\": \"yes or no\",\n\"explanations\": [\n    {{\n    \"bbox_2d\": [xx, xx, xx, xx], \"label\": \"xxx\"\n    }},\n    {{\n    \"bbox_2d\": [xx, xx, xx, xx], \"label\": \"xxx\"\n    }}]\n}}\n```\n<escapeShell \n\n具体问题:根据规则或识别要求，{describe}。 图中是否出现{event}？"}
        ],
    },
]

messages = image_messages

processor = AutoProcessor.from_pretrained(MODEL_PATH)
prompt = processor.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)

image_inputs, _, _ = process_vision_info(messages, return_video_kwargs=True)

mm_data = {}
if image_inputs is not None:
    mm_data["image"] = image_inputs

llm_inputs = {
    "prompt": prompt,
    "multi_modal_data": mm_data,
}

outputs = llm.generate([llm_inputs], sampling_params=sampling_params)
generated_text = outputs[0].outputs[0].text

print(generated_text)