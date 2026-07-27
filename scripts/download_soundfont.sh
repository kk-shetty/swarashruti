#!/bin/bash

URL="https://archive.org/download/fluidr3-gm-gs/FluidR3_GM_GS.sf2"
OUTPUT_DIR="models"
FILE_PATH="${OUTPUT_DIR}/FluidR3_GM.sf2"
MIN_SIZE=$((130 * 1024 * 1024)) # 130 MB in bytes

if [ -f "$FILE_PATH" ]; then
    echo "SoundFont file already exists at $FILE_PATH. Skipping download."
    exit 0
fi

mkdir -p "$OUTPUT_DIR"
curl --fail --location --progress-bar --output "$FILE_PATH" "$URL"

if [ $? -eq 0 ] && [ -f "$FILE_PATH" ] && [ $(wc -c < "$FILE_PATH") -ge $MIN_SIZE ]; then
    echo "Successfully downloaded the ${FILE_PATH} file from ${URL}."
    exit 0
else
    echo "Failed to download the ${FILE_PATH} file from ${URL}."
    rm -f "$FILE_PATH"
    echo "Partial or corrupted file removed."
    exit 1
fi