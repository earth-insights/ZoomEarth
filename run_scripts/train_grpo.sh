export CUDA_HOME=/root/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export DISABLE_FLASH_ATTN=1
export TRANSFORMERS_NO_FLASH_ATTENTION=1
export WANDB_MODE=offline
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
export REPO_HOME="${PROJECT_ROOT}"
echo "REPO_HOME: $REPO_HOME"
# TODO: change this to your own paths
data_paths="" 
image_folders=""
model_path=""

is_reward_customized_from_vlm_module=False
echo "data_paths: $data_paths"
echo "image_folders: $image_folders"

export EXP_NAME="LRS_GRO" # TODO: change this to your own experiment name
TASK_TYPE="LRS_GRO"
cd ${REPO_HOME}/src/train/RL/src/open-r1-multimodal

export DEBUG_MODE="true" # Enable Debug if you want to see the rollout of model during RL
# create the run directory and log file
mkdir -p ${REPO_HOME}/runs/${EXP_NAME}/log
export LOG_PATH="${REPO_HOME}/runs/${EXP_NAME}/log/debug_log.$(date +%Y-%m-%d-%H-%M-%S).txt"
# MAX_STEPS=1200 # TODO: change this to your own max steps

# export WANDB_DISABLED=true
torchrun --nproc_per_node="8" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12349" \
  src/open_r1/grpo_jsonl.py \
    --use_vllm False \
    --output_dir ${REPO_HOME}/checkpoints/rl/${EXP_NAME} \
    --resume_from_checkpoint True \
    --model_name_or_path $model_path \
    --data_file_paths $data_paths \
    --image_folders $image_folders \
    --is_reward_customized_from_vlm_module $is_reward_customized_from_vlm_module \
    --task_type $TASK_TYPE \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --gradient_checkpointing true \
    --logging_steps 1 \
    --num_train_epochs 1 \
    --bf16 \
    --attn_implementation sdpa \
    --run_name ${EXP_NAME} \
    --data_seed 42 \
    --save_steps 50 \
    --num_generations 4 \
    --max_completion_length 2048 \
    --reward_funcs iou format answer \
    --beta 0.04 \
    --report_to wandb \
    --dataset-name this_is_not_used \
    --learning_rate 1e-7

echo "Training completed for ${EXP_NAME}"
