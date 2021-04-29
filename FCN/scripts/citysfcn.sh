export NGPUS=3
CUDA_VISIBLE_DEVICES=0,1,2 python -m torch.distributed.launch --nproc_per_node=$NGPUS train.py --model fcn32s --backbone vgg16 --dataset citys --lr 0.0001 --epochs 80 --batch-size 16