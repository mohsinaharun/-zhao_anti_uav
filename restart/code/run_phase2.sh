#!/usr/bin/env bash
# Phase 2 sequence, exactly as pre-registered in restart/phase2/PREREG.md. Resumable
# (each run is skipped if its JSON exists).
set -u
cd "$(dirname "$0")/../.."
export GPU_SOFT_CAP=540 PYTHONUNBUFFERED=1
PY="$HOME/Documents/anti_uav/Scripts/python.exe"; R=restart/code/phase2_fl.py; O=restart/results/phase2
m() { $PY -c "import json;print(json.load(open('$O/$1.json'))['val_map50'])"; }
best() { $PY -c "import json,sys;v=sys.argv[2:];print(max(v,key=lambda x:json.load(open('$O/'+sys.argv[1].replace('X',x)+'.json'))['val_map50']))" "$@"; }

echo "== FL lr at mu=0 (no DP, seed 0)"
for lr in 0.01 0.03 0.1; do $PY $R --mode fl --lr $lr --mu 0 --seed 0 | grep final; done
LR=$(best fl_nodp_B256_s0_lrX_mu0_R12_seed0 0.01 0.03 0.1); echo "FL lr = $LR"
echo "== FedProx mu at lr=$LR"
for mu in 0.01 0.1 1; do $PY $R --mode fl --lr $LR --mu $mu --seed 0 | grep final; done
MU=$(best fl_nodp_B256_s0_lr${LR}_muX_R12_seed0 0 0.01 0.1 1); echo "FedProx mu = $MU"
echo "== FL+DP lr at sigma=0"
for lr in 0.01 0.03 0.1; do $PY $R --mode fl --dp --sigma 0 --lr $lr --mu $MU --seed 0 | grep final; done
DLR=$(best fl_dp_B256_s0_lrX_mu${MU}_R12_seed0 0.01 0.03 0.1); echo "FL+DP lr = $DLR"
echo "LR=$LR MU=$MU DLR=$DLR" > $O/selected.txt
echo "== 2x2, 10 paired seeds"
for s in 0 1 2 3 4 5 6 7 8 9; do
  $PY $R --mode fl --lr $LR --mu $MU --seed $s | grep final
  $PY $R --mode fl --dp --sigma 2 --lr $DLR --mu $MU --seed $s | grep final
  $PY $R --mode central --lr 0.03 --seed $s | grep final
  $PY $R --mode central --dp --sigma 2 --lr 0.03 --seed $s | grep final
done
echo "PHASE 2 RUNS DONE"
