# Run VLM-R1-OVD on Altlas 300I Duo by vllm-ascend

## Run docker container

First, download the image

```
docker pull quay.io/ascend/vllm-ascend:v0.10.0rc1-310p
```

Second, run the container
```yaml
# Update the vllm-ascend image
docker run --rm \
--name vllm-ascend \
--device /dev/davinci0 \
--device /dev/davinci_manager \
--device /dev/devmm_svm \
--device /dev/hisi_hdc \
-v /usr/local/dcmi:/usr/local/dcmi \
-v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
-v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
-v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
-v /etc/ascend_install.info:/etc/ascend_install.info \
-v /root/.cache:/root/.cache \
-e PYTORCH_NPU_ALLOC_CONF=max_split_size_mb:256 \
-p 8000:8000 \
-it quay.io/ascend/vllm-ascend:v0.10.0rc1-310p bash
```

Then, download the model `omlab/VLM-R1-Qwen2.5VL-3B-OVD-0321`

```
huggingface-cli download --resume-download omlab/VLM-R1-Qwen2.5VL-3B-OVD-0321 --local-dir VLM-R1-Qwen2.5VL-3B-OVD-0321
```

Last, we need change the `torch_dtype` from bfloat16 to float16 in config.yaml.
## Offline Inference
Run the following script [offline_inference.py](offline_inference.py) to execute offline inference on NPU:
```
from transformers import AutoProcessor
from vllm import LLM, SamplingParams
from qwen_vl_utils import process_vision_info

MODEL_PATH = "VLM-R1-Qwen2.5VL-3B-OVD-0321"

llm = LLM(
    model=MODEL_PATH,
    max_model_len=16384,
    limit_mm_per_prompt={"image": 10},
    dtype="float16", 
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
```

## Online Inference
Run the following command to start the vllm-ascend server:
```
vllm serve VLM-R1-Qwen2.5VL-3B-OVD-0321 \
  --max-model-len 16384 \
  --limit-mm-per-prompt '{"image": 10}' \
  --dtype float16 \
  --enforce-eager \
  --port 8000 \
  --host 0.0.0.0 \
  --trust-remote-code \
```
Once your server is started, you can send requests to the server using the following command:
```
curl -X POST "http://localhost:8000/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "VLM-R1-Qwen2.5VL-3B-OVD-0321",
    "messages": [
      {
        "role": "user",
        "content": [
            {
                "type": "image_url",
                "image_url": {
                    "url": "https://cbu01.alicdn.com/img/ibank/2018/960/214/9170412069_1052252572.jpg"  
                }
            },
            {
                "type": "text",
                "text": "请分析图像并回答以下问题。您的回答应包含对图像内容的简要描述和最终答案。描述使用 `<description></description>` 标签包裹，答案使用 `</think></think>` 标签包裹。答案必须以 JSON 格式输出，包含 \"yes\" 或 \"no\"，并提供相关物体的边界框坐标作为解释。如果没有涉及具体物体，则将 \"explanations\" 设为 \"None\"。输出格式如下：\n\n<description>对图像内容的简要描述写在这里</description>\n\n</think>\n```json\n{{\"answer\": \"yes or no\",\n\"explanations\": [\n    {{\n    \"bbox_2d\": [xx, xx, xx, xx], \"label\": \"xxx\"\n    }},\n    {{\n    \"bbox_2d\": [xx, xx, xx, xx], \"label\": \"xxx\"\n    }}]\n}}\n```\n<|FunctionCallEnd|>\n\n具体问题:根据规则或识别要求，{describe}。 图中是否出现{event}？"
            }  
        ]
      }
    ]
  }'
```