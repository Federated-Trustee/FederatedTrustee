#!/usr/bin/env bash
set -euo pipefail

# Keep Flower/Ray outputs readable by suppressing noisy dependency warnings.
export PYTHONWARNINGS="ignore::DeprecationWarning,ignore:The given NumPy array is not writable:UserWarning"


CONFIG_PATH="${1:-configs/nsl_kdd.yaml}"
ROUND_MODE="${2:-final}"

echo "=========================================="
echo "SHERPAxTRUSTEE pipeline"
echo "=========================================="
echo "Config path : ${CONFIG_PATH}"
echo "Round mode  : ${ROUND_MODE}"
echo "=========================================="

if [[ ! -f "${CONFIG_PATH}" ]]; then
    echo "[ERROR] Config file not found: ${CONFIG_PATH}"
    exit 1
fi

python main.py --config "${CONFIG_PATH}"

case "${ROUND_MODE}" in
    final)
        python scripts/analyze_model_agreement.py --config "${CONFIG_PATH}"
        python scripts/analyze_trustee_agreement.py --config "${CONFIG_PATH}"
        ;;
    all-rounds)
        python scripts/analyze_model_agreement.py --config "${CONFIG_PATH}" --all-rounds
        python scripts/analyze_trustee_agreement.py --config "${CONFIG_PATH}" --all-rounds
        ;;
    *)
        echo "[ERROR] Invalid round mode: ${ROUND_MODE}"
        echo "Usage: bash run_pipeline.sh [CONFIG_PATH] [final|all-rounds]"
        exit 1
        ;;
esac

echo "=========================================="
echo "[DONE] Pipeline completed successfully."
echo "=========================================="