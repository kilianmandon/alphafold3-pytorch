#!/bin/bash

cd /workspace
curl -L https://github.com/peak/s5cmd/releases/download/v2.3.0/s5cmd_2.3.0_Linux-64bit.tar.gz | tar -xz
sudo mv s5cmd /usr/local/bin/


git clone https://github.com/kilianmandon/alphafold3-pytorch.git
cd alphafold3-pytorch

uv sync

uv pip install --upgrade 'torch>=2.9'
uv pip install --upgrade torchvision
uv pip install --upgrade lightning

.venv/bin/python image_diffusion/training.py --config "$config_path" 2>&1 | tee training.log

# upload latest checkpoint to DO space
filename="diffusion_$(date +'%Y-%m-%d_%H-%M-%S').ckpt"
s5cmd --endpoint-url https://af3-data.tor1.digitaloceanspaces.com cp image_diffusion/checkpoints/last.ckpt s3://data/checkpoints/$filename

vastai destroy instance $instance_id
