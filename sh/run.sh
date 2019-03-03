#!/bin/bash

# top-down view 
TOP_PATH=/scratch1/sniu/pc

# pointnet
python main.py --loss_type=emd --save_path=${TOP_PATH}/pointnet_full_emd \
               --fb_split=False --PW=32 --PL=32 --cell_max_points=512 \
               --cell_min_points=50 --encoder=pointnet --decoder=pointnet \
               --stacked=False --rotation=False --pooling=max

# # dgcnn
# python main.py --loss_type=emd --save_path=${TOP_PATH}/dgcnngrid_full_emd \
#                --fb_split=False --PW=32 --PL=32 --cell_max_points=512 \
#                --cell_min_points=50 --encoder=dgcnn --decoder=pointgrid \
#                --stacked=False --batch_size=128 --rotation=False --pooling=max

# # foldingnet
# python main.py --loss_type=emd --save_path=${TOP_PATH}/pointgrid_full_emd \
#                --fb_split=False --PW=32 --PL=32 --cell_max_points=512 \
#                --cell_min_points=30 --encoder=pointnet --decoder=pointgrid \
#                --stacked=False

# # stacked network
# python main.py --loss_type=chamfer --save_path=${TOP_PATH}/stackpointnet_full_chamfer \
#                --fb_split=False --PW=16,32 --PL=16,32 --cell_max_points=512,512 \
#                --cell_min_points=30,30 --encoder=pointnet --decoder=pointnet \
#                --stacked=True

# python main.py --loss_type=emd --save_path=${TOP_PATH}/stackpointnet_full_emd \
#                --fb_split=False --PW=16,32 --PL=16,32 --cell_max_points=512,512 \
#                --cell_min_points=30,30 --encoder=pointnet --decoder=pointnet \
#                --stacked=True

# ours
python main.py --loss_type=emd --save_path=${TOP_PATH}/inception_full_emd_add \
               --fb_split=False --PW=32 --PL=32 --cell_max_points=512 \
               --cell_min_points=50 --encoder=inception --decoder=pointnet \
               --stacked=False --rotation=True --pooling=mean --batch_size=128 --training=True --range_view=False

python main.py --loss_type=emd --save_path=${TOP_PATH}/inception_full_emd_add \
               --fb_split=False --PW=32 --PL=32 --cell_max_points=512 \
               --cell_min_points=50 --encoder=inception --decoder=pointnet \
               --stacked=False --rotation=True --pooling=mean --batch_size=128 --training=False --range_view=False

