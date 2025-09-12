from huggingface_hub import snapshot_download

cache_dir = snapshot_download(
    repo_id="Qwen/Qwen2-VL-7B-Instruct",
    local_files_only=True  # 只查本地，不联网
)
print(cache_dir)