#!/bin/bash

set -e

echo "🔍 Checking dependencies..."
python check_dependencies.py

echo "📸 Running COLMAP + conversion..."
python convert.py --source ./data/images --output ./output --colmap_path $(which colmap)

echo "🏃 Starting 3DGS training..."
python train.py -s ./output --iterations 30000

echo "📽️ Rendering result..."
python render.py -m ./output

echo "✅ All done!"

# 执行权限
# chmod +x run_all.sh
# ./run_all.sh