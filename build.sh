#!/usr/bin/env bash
# Exit immediately if a command exits with a non-zero status
set -o errexit

# Step 3: Install Python dependencies
echo "Installing Python dependencies..."
pip install -r requirements.txt

# Step 6: Install ffmpeg
echo "Downloading and extracting static ffmpeg..."
wget https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz
tar -xf ffmpeg-release-amd64-static.tar.xz

# Move the binaries to a location in your PATH so your app can use them
mkdir -p ~/.local/bin
mv ffmpeg-*-amd64-static/ffmpeg ~/.local/bin/
mv ffmpeg-*-amd64-static/ffprobe ~/.local/bin/

# Clean up the downloaded archive to save space
rm -rf ffmpeg-release-amd64-static.tar.xz ffmpeg-*-amd64-static
echo "Build complete!"