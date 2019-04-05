#!/bin/bash

# top-down view 
TOP_PATH=/scratch4/sniu/pc

# pointnet
# python main.py --loss_type=emd --save_path=${TOP_PATH}/test_pointnet_multi_3 \
#                --fb_split=False --PW=32,16,8 --PL=32,16,8 --cell_max_points=512,512,512 \
#                --cell_min_points=50,50,50 --encoder=pointnet --decoder=pointnet \
#                --stacked=False --rotation=False --pooling=max --training=True --num_epochs=281

# # evaluation
# python main.py --loss_type=emd --save_path=${TOP_PATH}/test_pointnet_multi_3 \
#                --fb_split=False --PW=32,16,8 --PL=32,16,8 --cell_max_points=512,512,512 \
#                --cell_min_points=50,50,50 --encoder=pointnet --decoder=pointnet \
#                --stacked=False --rotation=False --pooling=max --training=False --weights="model-280.ckpt"

#### this is up version: 64,32 -> 32
python main.py --loss_type=emd --save_path=${TOP_PATH}/pointnet_multi_new \
               --fb_split=False --PW=64,32 --PL=64,32 --cell_max_points=512,512 \
               --cell_min_points=50,50 --encoder=pointnet --decoder=pointnet \
               --stacked=False --rotation=False --pooling=max --training=True --num_epochs=281 --combination=up

# evaluation
python main.py --loss_type=emd --save_path=${TOP_PATH}/pointnet_multi_new \
               --fb_split=False --PW=64,32 --PL=64,32 --cell_max_points=512,512 \
               --cell_min_points=50,50 --encoder=pointnet --decoder=pointnet \
               --stacked=False --rotation=False --pooling=max --training=False --weights="model-280.ckpt" --combination=up
               
### this is down version: 32,16 -> 32
python main.py --loss_type=emd --save_path=${TOP_PATH}/pointnet_multi_new \
               --fb_split=False --PW=32,16 --PL=32,16 --cell_max_points=512,512 \
               --cell_min_points=50,50 --encoder=pointnet --decoder=pointnet \
               --stacked=False --rotation=False --pooling=max --training=True --num_epochs=281 --combination=down

# evaluation
python main.py --loss_type=emd --save_path=${TOP_PATH}/pointnet_multi_new \
               --fb_split=False --PW=32,16 --PL=32,16 --cell_max_points=512,512 \
               --cell_min_points=50,50 --encoder=pointnet --decoder=pointnet \
               --stacked=False --rotation=False --pooling=max --training=False --weights="model-280.ckpt" --combination=down
               
               
