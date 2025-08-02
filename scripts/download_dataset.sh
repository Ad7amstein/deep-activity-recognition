#!/bin/bash
echo -e "\nDownloading Dataset in Progress..."

DATASET_LINK="https://www.kaggle.com/api/v1/datasets/download/ahmedmohamed365/volleyball"
DEST_DIR="data/volleyball.zip"
EX_DIR="data/volleyball"

echo -e "\n✅Downloading Dataset Completed Successfully"

curl -L --create-dirs -o "$DEST_DIR"\
  "$DATASET_LINK"

echo -e "\nUnzipping Files"

unzip "$DEST_DIR" -d "$EX_DIR"

echo -e "\n✅Done Unzipping Files"