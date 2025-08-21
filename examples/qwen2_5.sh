#!/bin/bash
#SBATCH --job-name=Qwen2.5Math1.5B
#SBATCH -p a100                           # 使用 a100 分区
#SBATCH --gres=gpu:8                      # 请求 8 张 GPU
#SBATCH -C a100_80                        # 要求 A100 80GB 显存
#SBATCH -N 1                              # 单节点
#SBATCH -n 1                              # 单任务
#SBATCH -t 06:00:00                       # 最大运行时间
#SBATCH -o /home/hpc/b273dd/b273dd11/EasyR1/examples/outputs/%x.%j.out              # 输出日志
#SBATCH -e /home/hpc/b273dd/b273dd11/EasyR1/examples/outputs/%x.%j.err              # 错误日志

set -x

# 环境设置
eval "$(mamba shell hook --shell bash)"
mamba activate --prefix ~/miniforge3/envs/easy    # 激活你的 conda 环境（根据实际名称）

# 取消显式 GPU 可见性限制
unset ROCR_VISIBLE_DEVICES

# # 数据路径
# gsm8k_train_path=$HOME/data/gsm8k/train.parquet
# gsm8k_test_path=$HOME/data/gsm8k/test.parquet

# # 训练与测试数据文件
# train_files="$gsm8k_train_path"
# test_files="$gsm8k_test_path"


export PYTHONUNBUFFERED=1
export WANDB_API_KEY=hf_XXXXXXXXXXXXXXXX
export HUGGINGFACE_HUB_TOKEN=hf_XXXXXXXXXXXXXXXX
export http_proxy=http://proxy.nhr.fau.de:80
export https_proxy=http://proxy.nhr.fau.de:80

MODEL_PATH=Qwen/Qwen2.5-Math-1.5B-Instruct  # replace it with your local file path

python3 -m verl.trainer.main \
    config=/home/hpc/b273dd/b273dd11/EasyR1/examples/config.yaml \
    worker.actor.model.model_path=${MODEL_PATH}
