vllm serve VLM-R1-Qwen2.5VL-3B-OVD-0321 \
  --max-model-len 16384 \
  --enforce-eager \
  --port 8000 \
  --host 0.0.0.0 \
  --trust-remote-code \