#export CNCL_MEM_POOL_MULTI_CLIQUE_ENABLE=1
#export CNCL_DIE_BUFFER_ENABLE=0
python3.8 \
          -m paddle.distributed.launch --mlus="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15" tools/train.py \
          -c ./ppcls/configs/ImageNet/Inception/InceptionV4.yaml \
          -o Global.device=mlu \
          -o Global.seed=1234 \
          -o DataLoader.Train.sampler.batch_size=80 \
	  -o Global.use_visualdl=True
