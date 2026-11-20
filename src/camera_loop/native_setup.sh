#!/usr/bin/env bash
# Invoked inside an isolated Conda prefix by reaim.
set -euo pipefail
model=$1
repo=$2
package_root=$3
cd "$repo"
export MAX_JOBS="${MAX_JOBS:-4}"
export CUDA_HOME="${CONDA_PREFIX}"
pip=(python -m pip)
case "$model" in
  observer)
    "${pip[@]}" install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
    "${pip[@]}" install -e "$repo" opencv-python-headless scipy
    ;;
  sana)
    SANA_SKIP_TE=1 bash environment_setup.sh
    ;;
  matrixgame)
    "${pip[@]}" install torch==2.10.0 torchvision==0.25.0 xformers==0.0.34 --index-url https://download.pytorch.org/whl/cu128
    "${pip[@]}" install -r requirements.txt
    ;;
  dreamx)
    "${pip[@]}" install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
    "${pip[@]}" install ninja packaging wheel setuptools
    "${pip[@]}" install --no-build-isolation -r requirements.txt
    ;;
  worldplay)
    "${pip[@]}" install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
    "${pip[@]}" install -r requirements.txt
    "${pip[@]}" install --no-build-isolation flash-attn
    "${pip[@]}" install -e "$4"
    ;;
  worldwarp)
    "${pip[@]}" install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu126
    "${pip[@]}" install ninja packaging wheel setuptools
    "${pip[@]}" install --no-build-isolation flash-attn 'git+https://github.com/facebookresearch/pytorch3d.git' ./src/fused-ssim ./src/simple-knn
    "${pip[@]}" install -r requirements.txt
    (cd src/ttt3r/croco/models/curope && python setup.py build_ext --inplace)
    ;;
  lyra)
    cd Lyra-2
    export CC="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc"
    export CXX="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++"
    "${pip[@]}" install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
    site=$(python -c 'import site; print(site.getsitepackages()[0])')
    export CPATH="$CUDA_HOME/include:$site/nvidia/cudnn/include:$site/nvidia/nccl/include:${CPATH:-}"
    export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$site/torch/lib:$site/nvidia/cuda_runtime/lib:$site/nvidia/cudnn/lib:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
    "${pip[@]}" install --no-deps -r requirements.txt
    "${pip[@]}" install 'git+https://github.com/microsoft/MoGe.git'
    "${pip[@]}" install --no-build-isolation 'transformer_engine[pytorch]'
    test -e "$site/nvidia/cudart" || ln -s "$site/nvidia/cuda_runtime" "$site/nvidia/cudart"
    "${pip[@]}" install --no-build-isolation --no-binary flash-attn flash-attn==2.6.3
    "${pip[@]}" install --no-build-isolation -e lyra_2/_src/inference/vipe -e 'lyra_2/_src/inference/depth_anything_3[gs]'
    ;;
  *) exit 2 ;;
esac
# Make the controller importable from subprocesses which reset PYTHONPATH.
python - "$package_root" <<'PY'
from pathlib import Path
import site, sys
dest = Path(site.getsitepackages()[0]) / 'camera_loop'
source = Path(sys.argv[1]) / 'camera_loop'
if dest.is_symlink() and dest.resolve() == source.resolve():
    sys.exit(0)
dest.symlink_to(source, target_is_directory=True)
PY
