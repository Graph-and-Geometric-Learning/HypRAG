CHECKPOINT_DIR="src/contrastors/ckpts/albert_mlm_rotary"
latest_ckpt=$(ls -t ${CHECKPOINT_DIR}/ 2>/dev/null | head -n 1)
latest_ckpt="${CHECKPOINT_DIR}/${latest_ckpt}"

if [ -n "$latest_ckpt" ]; then
    echo "Found latest checkpoint: ${latest_ckpt}"
else
    echo "No checkpoint found. Starting from scratch."
fi