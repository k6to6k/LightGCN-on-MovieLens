#!/bin/bash

# 设置GPU
export CUDA_VISIBLE_DEVICES=7  # 使用7号GPU
GPU_ID=0  # 对应CUDA_VISIBLE_DEVICES中的索引

# 检查是否有.best_model_path文件，从而获取模型路径
if [ -f .best_model_path ]; then
    MODEL_PATH=$(cat .best_model_path)
else
    echo "Error: .best_model_path file not found."
    echo "Please run train.sh first to generate a best model."
    exit 1
fi

# 结果输出目录
OUTPUT_DIR="./results/final_evaluation"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
MODEL_NAME=$(basename $(dirname ${MODEL_PATH}))
OUTPUT_FILE="${OUTPUT_DIR}/eval_${MODEL_NAME%.*}_${TIMESTAMP}.json"  # 结果输出文件

# 模型参数
EMBED_DIM=64
N_LAYERS=3

# 评估参数
K=20
BATCH_SIZE=256

# 创建输出目录
mkdir -p ${OUTPUT_DIR}

echo "Starting evaluation on GPU ${CUDA_VISIBLE_DEVICES}"
echo "Model path: ${MODEL_PATH}"
echo "Output file: ${OUTPUT_FILE}"

# 运行评估
python eval.py \
  --model_path ${MODEL_PATH} \
  --data_path ${DATA_PATH} \
  --output_path ${OUTPUT_FILE} \
  --embed_dim ${EMBED_DIM} \
  --n_layers ${N_LAYERS} \
  --k ${K} \
  --batch_size ${BATCH_SIZE} \
  --gpu ${GPU_ID}

echo "Evaluation completed! Results saved to ${OUTPUT_FILE}"
