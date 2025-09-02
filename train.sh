#!/bin/bash

# 指定GPU号码
export CUDA_VISIBLE_DEVICES=7

# --- 阶段一（续）：专注核心参数区域 ---

# --- 固定参数 ---
EMBED_DIM=64
N_LAYERS=3
BATCH_SIZE=4096
EPOCHS=150 # 增加Epochs以确保充分收敛
SEED=42
EVAL_FREQ=5
EARLY_STOP="--early_stop"

# --- 调参范围 (最有希望的组合) ---
# L2正则化强度
WEIGHT_DECAY_OPTIONS=(5e-5 1e-4)
# 学习率
LR_OPTIONS=(0.005 0.01)

echo "--- Continuing Stage 1 Grid Search: Focusing on Promising Parameters ---"

for wd in "${WEIGHT_DECAY_OPTIONS[@]}"; do
  for lr in "${LR_OPTIONS[@]}"; do
    
    # --- 动态路径参数 ---
    TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
    EXP_NAME="lightgcn_wd${wd}_lr${lr}"
    RUN_DIR="runs/${EXP_NAME}" # 结果继续保存在runs/目录下
    CHECKPOINT_DIR="${RUN_DIR}/checkpoints"
    RESULT_DIR="${RUN_DIR}/results"
    DATA_PATH="./data.pt"
    BEST_MODEL_PATH="${RUN_DIR}/best_model.pt"

    # 创建必要的目录
    mkdir -p ${CHECKPOINT_DIR} ${RESULT_DIR}

    # --- 运行训练 ---
    echo ""
    echo "==============================================="
    echo "Running experiment: wd=${wd}, lr=${lr}"
    echo "Results will be saved in: ${RUN_DIR}"
    echo "==============================================="

    python main.py \
      --seed ${SEED} \
      --lr ${lr} \
      --epochs ${EPOCHS} \
      --batch_size ${BATCH_SIZE} \
      --embed_dim ${EMBED_DIM} \
      --n_layers ${N_LAYERS} \
      --weight_decay ${wd} \
      --eval_freq ${EVAL_FREQ} \
      --checkpoint_dir ${CHECKPOINT_DIR} \
      --result_dir ${RESULT_DIR} \
      --data_path ${DATA_PATH} \
      --best_model_path ${BEST_MODEL_PATH} \
      ${EARLY_STOP}

  done
done

echo "--- Focused Grid Search Completed! ---"

# --- 自动汇总所有实验结果 ---
echo ""
echo "--- Summarizing all experiment results ---"
SUMMARY_FILE="runs/final_summary_report.txt"
if [ -f "$SUMMARY_FILE" ]; then
    rm "$SUMMARY_FILE" # 删除旧的报告以重新生成
fi

# 查找所有summary.txt文件并追加到总报告中
find runs -name "summary.txt" | sort | while read -r summary_file; do
    echo "Appending results from: $summary_file"
    cat "$summary_file" >> "$SUMMARY_FILE"
    echo -e "\n\n" >> "$SUMMARY_FILE" # 添加分隔符
done

echo "--- All results have been summarized in: ${SUMMARY_FILE} ---"
echo "You can now view this file to compare all runs."