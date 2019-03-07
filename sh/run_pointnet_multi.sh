#!/bin/bash

# top-down view 
TOP_PATH=/scratch4/sniu/pc

# pointnet
python main.py --loss_type=emd --save_path=${TOP_PATH}/pointnet_multi_emd \
               --fb_split=False --PW=32 --PL=32 --cell_max_points=512 \
               --cell_min_points=50 --encoder=pointnet --decoder=pointnet \
               --stacked=False --rotation=False --pooling=max --training=True --num_epochs=281

# evaluation
python main.py --loss_type=emd --save_path=${TOP_PATH}/pointnet_multi_emd \
               --fb_split=False --PW=32 --PL=32 --cell_max_points=512 \
               --cell_min_points=50 --encoder=pointnet --decoder=pointnet \
               --stacked=False --rotation=False --pooling=max --training=False --weights="model-280.ckpt"
