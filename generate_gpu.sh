export OPENAI_LOGDIR=/data/guozhanqiang/binary2gray/Improve-DDPM
export CUDA_VISIBLE_DEVICES=0,1,2,3,4
mpiexec -n 5 python scripts/image_train_3D.py
mpiexec -n 3 python scripts/image_finetune.py