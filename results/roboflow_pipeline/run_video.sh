#!/usr/bin/env bash
# Video pipeline: sample -> seg prelabel -> zip (upload вручную). Дата: 2026-05-26
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PIPELINE="$ROOT/results/roboflow_pipeline"
SCRIPTS="$PIPELINE/scripts"
DATE_TAG="$(date +%Y%m%d)"
BATCH_NAME="tt_img0291_v2_${DATE_TAG}"
RUN_DIR="$PIPELINE/runs/img0291_v2_${DATE_TAG}"

mkdir -p "$RUN_DIR"

echo "=== Roboflow video pipeline v2 (IMG_0291.MOV, gameplay 45-180s) ==="
echo "ROOT=$ROOT"
echo "RUN_DIR=$RUN_DIR"
echo "BATCH=$BATCH_NAME"
echo

cd "$SCRIPTS"

echo "[1/3] Sample frames (45-180s, stratified, require-table-ball, count=80)..."
python3 sample_frames.py \
  --video sources/IMG_0291.MOV \
  --output "$RUN_DIR"

echo
echo "[2/3] Pre-label with train12_ball68 (seg polygons + Ball bbox, spatial players)..."
python3 prelabel_yolo_seg.py --run-dir "$RUN_DIR"

echo
echo "[3/3] Build upload zip (seg format, 5 classes)..."
python3 build_upload_zip.py --run-dir "$RUN_DIR" --format seg

echo
echo "=== Video pipeline complete (zip ready, upload manual) ==="
echo "Run dir: $RUN_DIR"
echo "Manifest: $RUN_DIR/manifest.json"
echo "Report:   $RUN_DIR/prelabel_report.json"
echo "Zip:      $RUN_DIR/upload.zip"
echo
echo "Upload (only after seg project exists in Roboflow UI):"
echo "  cd $SCRIPTS"
echo "  python3 upload_batch.py --run-dir $RUN_DIR --batch-name $BATCH_NAME --format seg --no-rebuild"
echo
echo "Label format (merged labels/):"
echo "  seg 0-4: TableLeft, TableRight, Net, Player1, Player2 (polygons)"
echo "  det 5:   Ball cx cy w h (excluded from Roboflow seg upload zip)"
echo
echo "Player classes: spatial by bbox center x vs net split_x (ignore train12 cls 2/3)."
