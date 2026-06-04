#!/bin/bash
# Download pre-exported RT-DETR ONNX model from GitHub Releases

set -e

# Configuration - Update these when creating GitHub release
GITHUB_REPO="fenggroup/vehicle-overtaking-tracker"
RELEASE_TAG="v1.0.0"
MODEL_FILE="rtdetr.onnx"
DOWNLOAD_URL="https://github.com/${GITHUB_REPO}/releases/download/${RELEASE_TAG}/${MODEL_FILE}"

if [ -f "${MODEL_FILE}" ]; then
    echo "[WARNING] ${MODEL_FILE} already exists!"
    read -p "Overwrite? (y/n): " OVERWRITE
    if [ "${OVERWRITE}" != "y" ] && [ "${OVERWRITE}" != "Y" ]; then
        exit 0
    fi
fi

if command -v curl &> /dev/null; then
    curl -L -o "${MODEL_FILE}" "${DOWNLOAD_URL}"
elif command -v wget &> /dev/null; then
    wget -O "${MODEL_FILE}" "${DOWNLOAD_URL}"
else
    echo "[ERROR] Neither curl nor wget found!"
    echo "Please install curl or wget, or download manually from:"
    echo "${DOWNLOAD_URL}"
    exit 1
fi

if [ ! -f "${MODEL_FILE}" ]; then
    echo "[ERROR] Model file not found after download!"
    exit 1
fi

SIZE=$(stat -f%z "${MODEL_FILE}" 2>/dev/null || stat -c%s "${MODEL_FILE}" 2>/dev/null)

if [ "${SIZE}" -lt 100000000 ]; then
    echo "[ERROR] File size too small (${SIZE} bytes, expected ~169MB)"
    exit 1
fi

echo "Download complete: ${MODEL_FILE} (${SIZE} bytes)"
