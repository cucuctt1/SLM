#!/usr/bin/env bash
set -e

echo "Installing required libraries (without reinstalling torch)..."
pip install -q kaggle tqdm numpy

mkdir -p /content/data
mkdir -p /content/checkpoints
mkdir -p /content/tokenizer

cat <<'EOF'
Setup complete.
Next steps in Colab:
1) Set Kaggle creds:
   import os
   os.environ['KAGGLE_USERNAME'] = 'YOUR_USERNAME'
   os.environ['KAGGLE_KEY'] = 'YOUR_KEY'
2) Run training:
   !python train.py
3) Run generation:
   !python generate.py
EOF
