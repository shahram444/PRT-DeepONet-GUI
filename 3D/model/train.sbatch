#!/bin/bash
# =============================================================================
# train.sbatch — train the 3D PRT-DeepONet.
#
#   sbatch train.sbatch                       # geodesic (the real model)
#   sbatch train.sbatch edt                   # ablation: Euclidean distance
#   sbatch train.sbatch none                  # ablation: no distance at all
#   sbatch train.sbatch gdf --with-velocity   # flow-conditioned variant
#
# The three distance runs are the ablation the paper rests on: they show that
# the GEODESIC field, not merely any distance field, is what buys the accuracy.
# -----------------------------------------------------------------------------
# PARTITION: set this to your lab's buy-in partition.  Find its name with:
#     sinfo -s                       # every partition you can see
#     sacctmgr show assoc user=$USER format=partition   # the ones you may use
# On a Meile-lab buy-in node that is a group partition, not the shared gpu_p.
# If the node has no GPU, the model is small enough (1.66M parameters, ~0.4 GB
# at batch 8) to train on CPU; just delete the --gres line and raise --time.
# =============================================================================
#SBATCH --job-name=prt3d_train
#SBATCH --partition=MEILE_LAB_PARTITION    # <-- set this
#SBATCH --gres=gpu:1                       # delete this line if the node is CPU-only
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8                  # DataLoader workers
#SBATCH --mem=48gb
#SBATCH --time=12:00:00
#SBATCH --output=train_%j.out
#SBATCH --error=train_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=YOUR_MYID@uga.edu

cd $SLURM_SUBMIT_DIR

# Find the real module name first:  ml spider PyTorch
ml purge
ml PyTorch                                 # substitute what ml spider actually shows
# If PyTorch is not a module on the lab node, use a conda/venv instead:
#   source ~/envs/prt3d/bin/activate

DIST=${1:-gdf}                             # gdf | edt | none
shift 2>/dev/null || true                  # anything else passes through to train.py

DATA=/scratch/$USER/prt3d/dataset/prt3d_dataset.h5
OUT=/scratch/$USER/prt3d/runs/$DIST

if [ ! -f "$DATA" ]; then
  echo "No dataset at $DATA. Run collect.py first."; exit 1
fi
mkdir -p "$OUT"

command -v nvidia-smi >/dev/null && nvidia-smi   # record which GPU you got, if any
python3 -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

python3 train.py \
    --data "$DATA" \
    --out  "$OUT" \
    --distance "$DIST" \
    --n-points 8192 \
    --batch-size 8 \
    --epochs 300 \
    --workers ${SLURM_CPUS_PER_TASK:-4} \
    "$@"

echo "done. summary: $OUT/summary.json   checkpoint: $OUT/best.pt"
