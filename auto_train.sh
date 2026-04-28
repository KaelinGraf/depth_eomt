#!/bin/bash
# auto_train.sh — restarts training from latest checkpoint on crash

CKPT_DIR="/home/kaelin/BinPicking/eomt/eomt"
CONFIG="configs/dinov3/occlusion_bp/panoptic/eomt_large_640.yaml"

while true; do
    # Find the latest checkpoint
    LATEST=$(find "$CKPT_DIR" -name "*.ckpt" -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2)
    
    if [ -n "$LATEST" ]; then
        echo "Resuming from: $LATEST"
        python3 main.py fit -c "$CONFIG" --trainer.devices 1 --ckpt_path "$LATEST"
    else
        echo "No checkpoint found, starting fresh"
        python3 main.py fit -c "$CONFIG" --trainer.devices 1
    fi
    
    EXIT_CODE=$?
    if [ $EXIT_CODE -eq 0 ]; then
        echo "Training completed successfully"
        break
    fi
    
    echo "Training crashed (exit code $EXIT_CODE). Restarting in 10s..."
    sleep 10
done
