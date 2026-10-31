#!/usr/bin/env bash
set -euo pipefail
root=/home/wave/MOL_growth_recalculation_20260830/steric_mixed_solvent_20260912_v1
bundle=/home/wave/MOL_growth_recalculation_20260830
code="$root/solvent_competition_20260912_v1"
stage="${1:?Specify pilot or grid}"
if [[ "$stage" == pilot ]]; then
    workers=16
    extra=()
elif [[ "$stage" == grid ]]; then
    workers=96
    extra=(--pilot-output "$root/pilot")
else
    exit 2
fi
test ! -e "$root/$stage"
test ! -e "$root/${stage}.pid"
cd "$code"
nohup env PYTHONHASHSEED=0 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    /usr/bin/python3 -u run_mixed_campaign.py --root "$root" --bundle-root "$bundle" \
    --manifest "$code/${stage}_manifest.csv" --output "$root/$stage" \
    --stage "$stage" --workers "$workers" "${extra[@]}" \
    > "$root/${stage}.log" 2>&1 < /dev/null &
echo "$!" > "$root/${stage}.pid"
cat "$root/${stage}.pid"
