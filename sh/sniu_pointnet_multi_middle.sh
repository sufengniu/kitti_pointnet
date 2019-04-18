#!/bin/bash

# top-down view 
TOP_PATH=/scratch4/sniu/pc


#### this is up version: 64,32 -> 32
python main.py --loss_type=emd --save_path=${TOP_PATH}/pointnet_multi_mid \
               --fb_split=False --PW=64,32,16 --PL=64,32,16 --cell_max_points=512,512,512 \
               --cell_min_points=50,50,50 --encoder=pointnet --decoder=pointnet \
               --stacked=False --rotation=False --pooling=max --training=True --num_epochs=281 --combination=sandwich

# evaluation
python main.py --loss_type=emd --save_path=${TOP_PATH}/pointnet_multi_mid \
               --fb_split=False --PW=64,32,16 --PL=64,32,16 --cell_max_points=512,512,512 \
               --cell_min_points=50,50,50 --encoder=pointnet --decoder=pointnet \
               --stacked=False --rotation=False --pooling=max --training=False --weights="model-280.ckpt" --combination=sandwich
               



               
