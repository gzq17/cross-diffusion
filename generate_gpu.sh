export OPENAI_LOGDIR=/data/guozhanqiang/binary2gray/Improve-DDPM
export CUDA_VISIBLE_DEVICES=0,1,2,3,4
mpiexec -n 5 python scripts/image_train_3D.py
mpiexec -n 3 python scripts/image_finetune.py

pip3 install blobfile
sudo apt-get install openmpi-bin openmpi-dev
pip3 install mpi4py
sudo apt-get install openmpi-bin openmpi-common libopenmpi-dev
pip3 install SimpleITK
pip3 install scipy
export OPENAI_LOGDIR=/opt/tiger/gzq_data/Improve-DDPM
wget https://cloud.tsinghua.edu.cn/f/ad290ea6ecaf4a4f9ae9/?dl=1 -O CTA_data.zip
sudo apt-get install tmux
