set -v
set -x
set -e

# 禁用 NCCL 设置
NCCL_P2P_DISABLE=1
NCCL_IB_DISABLE=1

GPU=4,5,6,7,

# Train EVAE
CUDA_VISIBLE_DEVICES=${GPU} accelerate launch --num_processes 4 --main_process_port 7777 src/train.py \
    --config config/denoise_50.yaml \
    --task_id 2 \
    --output_dir exp/DENOISE50/EVAE/DEMO5 \
    --train_type Evae \
    --max_train_steps 50000

Train EVAE
# CUDA_VISIBLE_DEVICES=${GPU} accelerate launch --num_processes 4 --main_process_port 8888 src/train.py \
#     --config config/denoise_50_temp.yaml \
#     --task_id 2 \
#     --output_dir exp/DENOISE50/EVAE/DEMO4 \
#     --train_type Evae \
#     --max_train_steps 50000

# Train Diffusion
# GPU=5,6,
# CUDA_VISIBLE_DEVICES=${GPU} accelerate launch --num_processes 1 --main_process_port 7777 src/train.py \
#     --config config/denoise_50.yaml \
#     --task_id 2 \
#     --output_dir exp/DENOISE50/MOSD/DEMO1 \
#     --train_type Diffusion \
#     --max_train_steps 50000
