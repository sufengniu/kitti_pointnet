#!/bin/bash

# top-down view 
TOP_PATH=/scratch4/sniu/pc

# pointnet
python main.py --loss_type=emd --save_path=${TOP_PATH}/pointnet_multi_emd \
               --fb_split=False --PW=32,16 --PL=32,16 --cell_max_points=512,512 \
               --cell_min_points=50,50 --encoder=pointnet --decoder=pointnet \
               --stacked=False --rotation=False --pooling=max --training=True
