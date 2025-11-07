#!/bin/bash
set -e

# This script downloads all the necessary models.
# Run this script from the project root directory (ITRI-GraspGen).

MODELS_DIR="models"

# GraspGenModels
echo "Downloading GraspGenModels..."
if [ -d "$MODELS_DIR/GraspGenModels" ]; then
    echo "GraspGenModels already exists, skipping."
else
    git clone https://huggingface.co/adithyamurali/GraspGenModels "$MODELS_DIR/GraspGenModels"
fi

# SAM2Models
echo "Downloading SAM2Models..."
if [ -d "$MODELS_DIR/SAM2Models" ] && [ "$(ls -A $MODELS_DIR/SAM2Models)" ]; then
    echo "SAM2Models already exists and is not empty, skipping."
else
    mkdir -p "$MODELS_DIR/SAM2Models"
    cp Third_Party/sam2/checkpoints/download_ckpts.sh "$MODELS_DIR/SAM2Models/"
    (cd "$MODELS_DIR/SAM2Models" && bash ./download_ckpts.sh)
fi

# FoundationStereoModels
echo "Downloading FoundationStereoModels..."
if [ -d "$MODELS_DIR/FoundationStereoModels" ]; then
    echo "FoundationStereoModels already exists, skipping."
else
    # gdown downloads to the current directory, so we execute it inside the models dir
    (cd "$MODELS_DIR" && uv run gdown --folder https://drive.google.com/drive/folders/1VhPebc_mMxWKccrv7pdQLTvXYVcLYpsf)
    mv "$MODELS_DIR/pretrained_models" "$MODELS_DIR/FoundationStereoModels"
fi

# GroundingDINOModels
echo "Downloading GroundingDINOModels..."
if [ -f "$MODELS_DIR/GroundingDINOModels/groundingdino_swint_ogc.pth" ]; then
    echo "GroundingDINO model already exists, skipping."
else
    mkdir -p "$MODELS_DIR/GroundingDINOModels"
    wget -q https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth -P "$MODELS_DIR/GroundingDINOModels"
fi


echo "All models downloaded successfully."
