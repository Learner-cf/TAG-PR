#!/bin/bash

TIME_TAG=$(date +"%Y%m%d_%H%M%S")
OUTDIR="runs/${TIME_TAG}"
mkdir -p "${OUTDIR}"

echo "===================================="
echo "Running 7 experiments: ${TIME_TAG}"
echo "Logs -> ${OUTDIR}"
echo "===================================="

echo "[1/7] baseline"
python main.py --config config/config_baseline.yaml > "${OUTDIR}/baseline.log" 2>&1

echo "[2/7] full"
python main.py --config config/config_trvd_full.yaml > "${OUTDIR}/full.log" 2>&1

echo "[3/7] wo_residual"
python main.py --config config/config_wo_residual.yaml > "${OUTDIR}/wo_residual.log" 2>&1

echo "[4/7] wo_view"
python main.py --config config/config_wo_view.yaml > "${OUTDIR}/wo_view.log" 2>&1

echo "[5/7] wo_orth"
python main.py --config config/config_wo_orth.yaml > "${OUTDIR}/wo_orth.log" 2>&1

echo "[6/7] wo_cons"
python main.py --config config/config_wo_cons.yaml > "${OUTDIR}/wo_cons.log" 2>&1

echo "[7/7] wo_bridge"
python main.py --config config/config_wo_bridge.yaml > "${OUTDIR}/wo_bridge.log" 2>&1

echo "===================================="
echo "All experiments finished."
echo "Check logs in ${OUTDIR}"
echo "===================================="