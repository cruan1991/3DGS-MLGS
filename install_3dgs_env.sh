#!/bin/bash

# === Step 1: 创建干净的 Conda 环境 ===
echo "🧪 Creating conda env: 3dgs"
conda create -n 3dgs python=3.8.10 nomkl -y -c conda-forge
conda activate 3dgs || source activate 3dgs

# === Step 2: 安装 PyTorch（pip wheel 版） ===
echo "🔥 Installing PyTorch (cu116 pip wheel)"
pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu116

# === Step 3: 设置 CUDA 编译环境（根据你的实际路径调整） ===
echo "⚙️ Setting CUDA environment to /usr/local/cuda-11.8"
export CUDA_HOME=/usr/local/cuda-11.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# === Step 4: 安装常用依赖(conda + pip) ===
echo "📦 Installing tqdm, plyfile, opencv, joblib"
conda install -y scipy scikit-learn plyfile -c conda-forge
pip install tqdm opencv-python joblib

# === Step 5: 安装 submodules 模块（你需要在 3dgs 项目根目录下运行） ===
echo "🔧 Installing CUDA extensions"
pip install ./submodules/diff-gaussian-rasterization
pip install ./submodules/simple-knn
pip install ./submodules/fused-ssim

# === Step 6: 验证安装 ===
echo "✅ Verifying setup"
python -c "import torch, scipy, sklearn, diff_gaussian_rasterization; print('🚀 3DGS + COLMAP environment ready!')"