#!/bin/bash

# top-down view 
TOP_PATH=/scratch4/sniu/pc

# pointnet
python main.py --loss_type=emd --save_path=${TOP_PATH}/pointnet_full_emd_40 \
               --fb_split=False --PW=32 --PL=32 --cell_max_points=512 \
               --cell_min_points=40 --encoder=pointnet --decoder=pointnet \
               --stacked=False --rotation=False --pooling=max --training=True

python main.py --loss_type=emd --save_path=${TOP_PATH}/pointnet_full_emd_30 \
               --fb_split=False --PW=32 --PL=32 --cell_max_points=512 \
               --cell_min_points=30 --encoder=pointnet --decoder=pointnet \
               --stacked=False --rotation=False --pooling=max --training=True

python main.py --loss_type=emd --save_path=${TOP_PATH}/pointnet_full_emd_20 \
               --fb_split=False --PW=32 --PL=32 --cell_max_points=512 \
               --cell_min_points=20 --encoder=pointnet --decoder=pointnet \
               --stacked=False --rotation=False --pooling=max --training=True

python main.py --loss_type=emd --save_path=${TOP_PATH}/pointnet_full_emd_10 \
               --fb_split=False --PW=32 --PL=32 --cell_max_points=512 \
               --cell_min_points=10 --encoder=pointnet --decoder=pointnet \
               --stacked=False --rotation=False --pooling=max --training=True

# pointnet
python main.py --loss_type=emd --save_path=${TOP_PATH}/pointnet_full_emd_40 \
               --fb_split=False --PW=32 --PL=32 --cell_max_points=512 \
               --cell_min_points=40 --encoder=pointnet --decoder=pointnet \
               --stacked=False --rotation=False --pooling=max --training=False

python main.py --loss_type=emd --save_path=${TOP_PATH}/pointnet_full_emd_30 \
               --fb_split=False --PW=32 --PL=32 --cell_max_points=512 \
               --cell_min_points=30 --encoder=pointnet --decoder=pointnet \
               --stacked=False --rotation=False --pooling=max --training=False

python main.py --loss_type=emd --save_path=${TOP_PATH}/pointnet_full_emd_20 \
               --fb_split=False --PW=32 --PL=32 --cell_max_points=512 \
               --cell_min_points=20 --encoder=pointnet --decoder=pointnet \
               --stacked=False --rotation=False --pooling=max --training=False

python main.py --loss_type=emd --save_path=${TOP_PATH}/pointnet_full_emd_10 \
               --fb_split=False --PW=32 --PL=32 --cell_max_points=512 \
               --cell_min_points=10 --encoder=pointnet --decoder=pointnet \
               --stacked=False --rotation=False --pooling=max --training=False
