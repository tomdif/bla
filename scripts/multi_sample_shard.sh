#!/bin/bash
# Shard 200 GSM8K-test problems across 6 GPUs for multi_sample_critic_pal.py.
# Each GPU gets ~34 problems × 16 candidates.
# Usage: multi_sample_shard.sh CKPT_PATH OUTPUT_DIR
set -e
CKPT=$1
OUT=$2
N_PROBLEMS=${3:-200}
SHARDS=6
PER_SHARD=$((($N_PROBLEMS + $SHARDS - 1) / $SHARDS))
mkdir -p $OUT

for i in 0 1 2 3 4 5; do
  start=$(($i * $PER_SHARD))
  end=$(( ($i + 1) * $PER_SHARD ))
  if [ $end -gt $N_PROBLEMS ]; then end=$N_PROBLEMS; fi
  echo "GPU $i: problems $start..$end"
  nohup bash -c "CUDA_VISIBLE_DEVICES=$i python scripts/multi_sample_critic_pal_shard.py \
    --ckpt $CKPT --critic runs/phase10/critic.pt \
    --start $start --end $end --n-samples 16 \
    --output $OUT/shard_$i" > $OUT/shard_$i.log 2>&1 &
done
echo "launched 6 shards"
