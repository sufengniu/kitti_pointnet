#!/bin/bash

# top-down view 
TOP_PATH=/scratch4/sniu/pc

python main.py --loss_type=emd --save_path=${TOP_PATH}/pointnet_full_emd \
               --fb_split=False --PW=32 --PL=32 --cell_max_points=512 \
               --cell_min_points=50 --encoder=pointnet --decoder=pointnet \
               --stacked=False --rotation=False --pooling=max --training=False --compress_all_kitti=True
