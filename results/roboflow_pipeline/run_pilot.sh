#!/usr/bin/env bash
# Smoke test Roboflow pipeline без raw video. Дата: 2026-05-26
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PIPELINE="$ROOT/results/roboflow_pipeline"
SCRIPTS="$PIPELINE/scripts"
DATE_TAG="$(date +%Y%m%d)"
BATCH_NAME="tt_pilot_smoke_${DATE_TAG}"
RUN_DIR="$PIPELINE/runs/pilot_${DATE_TAG}"

mkdir -p "$RUN_DIR"

echo "=== Roboflow pipeline pilot ==="
echo "ROOT=$ROOT"
echo "RUN_DIR=$RUN_DIR"
echo "BATCH=$BATCH_NAME"
echo

cd "$SCRIPTS"

echo "[1/4] Sample 20 frames from dataset/train (frame_000*, Ball labels)..."
python3 sample_frames.py \
  --from-dataset \
  --dataset-dir dataset/train \
  --count 20 \
  --output "$RUN_DIR"

echo
echo "[2/4] Pre-label with train12_ball68 (detection bbox)..."
python3 prelabel_yolo_det.py --run-dir "$RUN_DIR"

echo
echo "[3/4] Build upload zip..."
python3 build_upload_zip.py --run-dir "$RUN_DIR"

echo
echo "[4/4] Upload to Roboflow batch $BATCH_NAME..."
python3 upload_batch.py --run-dir "$RUN_DIR" --batch-name "$BATCH_NAME"

echo
echo "=== Pilot complete ==="
echo "Run dir: $RUN_DIR"
echo "State:   $RUN_DIR/state.json"
echo
echo "Next manual steps:"
echo "  1. Open Roboflow project auto-label-lmrwb -> Annotate -> batch '$BATCH_NAME'"
echo "  2. Review pre-labels from train12 (YOLO detection upload)"
echo "  3. Run autolabel matrix:"
echo "       cd $SCRIPTS"
echo "       python3 run_autolabel_matrix.py --run-dir $RUN_DIR"
echo "  4. Compare M1 (auto-label-lmrwb-instant-1) vs M2 (sam3) in Roboflow UI"
echo "  5. If batch_id is null in state.json, copy batch id from Roboflow UI Annotate tab"
echo "  6. After human review, export version:"
echo "       python3 export_version.py --run-dir $RUN_DIR --version 7"
echo
echo "Optional health check:"
echo "  python3 -c \"from roboflow_client import projects_health; import json; print(json.dumps(projects_health('auto-label-lmrwb'), indent=2)[:800])\""
