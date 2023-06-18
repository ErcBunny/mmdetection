# conda create --name openmmlab python=3.8 -y
# conda activate openmmlab
# inside your conda env run this
pip3 install torch torchvision ipykernel
pip install -U openmim
mim install mmengine mmpretrain
mim install "mmcv>=2.0.0"
pip install -v -e .