#!/bin/bash

# top-down view 
TOP_PATH=/scratch4/sniu/pc

# pointnet
python main.py --loss_type=emd --save_path=${TOP_PATH}/pointnet_range_multi \
               --fb_split=False --PW=128,64 --PL=8,8 --cell_max_points=512,512 \
               --cell_min_points=50,50 --encoder=pointnet --decoder=pointnet \
               --stacked=True --rotation=False --pooling=max --training=True --num_epochs=281 --partition_mode=range

# evaluation
python main.py --loss_type=emd --save_path=${TOP_PATH}/pointnet_range_multi \
               --fb_split=False --PW=128,64 --PL=8,8 --cell_max_points=512,512 \
               --cell_min_points=50,50 --encoder=pointnet --decoder=pointnet \
               --stacked=True --rotation=False --pooling=max --training=False --partition_mode=range --weights="model-280.ckpt"
