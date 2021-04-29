export NGPUS=2
CUDA_VISIBLE_DEVICES=0,2 python -m torch.distributed.launch --nproc_per_node=$NGPUS eval.py --model fcn32s --backbone vgg16 --dataset citys