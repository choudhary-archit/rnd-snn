#!/usr/bin/env bash
# Reproduce the Python environment for SpikeYOLO inference (CPU, no training).
#   - upstream code  -> py/SpikeYOLO   (BICLab/SpikeYOLO, pinned commit)
#   - venv           -> py/.venv       (Python 3.10, torch 2.4.1 CPU)
#   - checkpoints    -> py/weights/{int_23M,bin_23M}/best.pt
set -euo pipefail
cd "$(dirname "$0")"

UPSTREAM_COMMIT=dc51f35d0b6e970cd157080c98157a41af4392f5

command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
UV="$(command -v uv || echo "$HOME/.local/bin/uv")"

if [ ! -d SpikeYOLO ]; then
  git clone https://github.com/BICLab/SpikeYOLO.git
  git -C SpikeYOLO checkout -q "$UPSTREAM_COMMIT"
fi

[ -d .venv ] || "$UV" venv --python 3.10 .venv
# torch < 2.6 keeps torch.load(weights_only=False) as default, which the pickled checkpoints need
"$UV" pip install --python .venv/bin/python \
  --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple \
  "torch==2.4.1" "torchvision==0.19.1"
"$UV" pip install --python .venv/bin/python \
  "numpy<2" "opencv-python-headless<4.11" matplotlib pillow pyyaml requests scipy tqdm pandas \
  seaborn psutil py-cpuinfo thop "timm==0.9.16" "spikingjelly==0.0.0.0.14" einops torchinfo \
  dill gdown onnx
"$UV" pip install --python .venv/bin/python --no-deps -e SpikeYOLO

mkdir -p weights/int_23M weights/bin_23M assets
# 23.1M, T=1, D=4 integer-valued checkpoint (Drive folder)
[ -f weights/int_23M/best.pt ] || .venv/bin/gdown --folder \
  "https://drive.google.com/drive/folders/1c5p09ZRCFeK1M5wH6zQduJltZalMzQkZ" -O weights/int_23M
# 23.1M, T=1, D=4 binary-inference checkpoint (for yolo_spikformer_bin.py)
[ -f weights/bin_23M/best.pt ] || .venv/bin/gdown "1YQ29eDUfmaze2jl_UREX4Zeb1u8tpHfl" -O weights/bin_23M/best.pt
# Optional, larger models:
#   69M int:   .venv/bin/gdown 1rmcUMJztbjFFbbVqW8xwgshKNZel1psZ -O weights/int_69M/best.pt
#   Gen1 23M:  .venv/bin/gdown 1PnrsYWSOrCjvfRpYng_hsTcv35pGsrHb -O weights/gen1_23M/best.pt
[ -f assets/bus.jpg ] || curl -sL -o assets/bus.jpg https://ultralytics.com/images/bus.jpg

echo "Done. Activate with: source $(pwd)/.venv/bin/activate"
