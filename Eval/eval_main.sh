PRETRAIN_DIR="checkpoints"

graph_size=(50 100)
methods=(hmtf mtpomo mvmoe rf-te)

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m'

for s in "${graph_size[@]}"; do
    for method in "${methods[@]}"; do
        ckpt_path="$PRETRAIN_DIR/$s/$method/last.ckpt"
        if [[ -f "$ckpt_path" ]]; then
            echo -e "${BLUE}python Eval.py --checkpoint=$ckpt_path --size=$s --device=cuda:0${NC}"
            python Eval.py --checkpoint="$ckpt_path" --size=$s  --device=cuda:0
        else
            echo -e "${RED}Checkpoint not found: $ckpt_path${NC}"
        fi
    done
done
