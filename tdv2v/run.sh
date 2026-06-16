#!/usr/bin/env bash
# Tier-1 head-to-head. ViT-S, identical data/compute; flags select the condition.
# Swap --data synthetic -> sceneflow/kitti (and wire data.py) for the real run.
set -e
COMMON="--data synthetic --img 224 --patch 16 --dim 384 --depth 12 --heads 6 \
        --bs 128 --lr 1e-4 --ema 0.996 --steps 20000 --warmup 500 \
        --knn-every 1000 --log-every 100 --n-classes 50 --anticollapse sigreg"

python train.py $COMMON --out runs/mono                                  # baseline: temporal only
python train.py $COMMON --crossview --out runs/xview                     # + cross-view completion
python train.py $COMMON --crossview --commut --out runs/commut           # + commutativity
python train.py $COMMON --crossview --shuffle-control --out runs/shuffle # CAUSAL CONTROL

echo "Decisive comparison: does runs/xview KNN >> runs/mono  AND  >> runs/shuffle ?"
