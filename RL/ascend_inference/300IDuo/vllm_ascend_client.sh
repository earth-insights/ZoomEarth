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