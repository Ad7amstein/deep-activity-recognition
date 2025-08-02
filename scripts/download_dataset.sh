#!/bin/bash

# Replace with your actual folder ID (not the full URL)
FOLDER_ID="1HMmuAuZ9zuGS8Va19ytXJtlfgQbILyu9"

# Optional: Set download directory
DEST_DIR="data/"

# Download entire folder
gdown --folder https://drive.google.com/drive/folders/$FOLDER_ID -O "$DEST_DIR"
