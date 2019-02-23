#!/bin/bash

# range view 
TOP_PATH=/scratch3/sniu/pc

# pointnet
python main.py --loss_type=emd --save_path=/scratch4/sniu/pc/pointnet_full_emd_range --fb_split=False --PW=32 --PL=32 --cell_max_points=512 --cell_min_points=50 --encoder=pointnet --decoder=pointnet --stacked=False --rotation=False --pooling=max --training=False --range_view=True

echo "-------------------------------"
# pointnet foldingnet
python main.py --loss_type=emd --save_path=/scratch4/sniu/pc/pointgrid_full_emd_range --fb_split=False --PW=32 --PL=32 --cell_max_points=512 --cell_min_points=50 --encoder=pointnet --decoder=pointgrid --stacked=False --rotation=False --pooling=max --training=False --range_view=True

echo "-------------------------------"
# dgcnn 
python main.py --loss_type=emd --save_path=${TOP_PATH}/dgcnn_full_emd_range --fb_split=False --PW=32 --PL=32 --cell_max_points=512 --cell_min_points=50 --encoder=dgcnn --decoder=pointnet --stacked=False --batch_size=128 --rotation=False --pooling=max --training=False --range_view=True

echo "-------------------------------"
# dgcnn foldingnet
python main.py --loss_type=emd --save_path=${TOP_PATH}/dgcnngrid_full_emd_range --fb_split=False --PW=32 --PL=32 --cell_max_points=512 --cell_min_points=50 --encoder=dgcnn --decoder=pointgrid --stacked=False --batch_size=128 --rotation=False --pooling=max --training=False --range_view=True

echo "-------------------------------"
# inception
python main.py --loss_type=emd --save_path=${TOP_PATH}/inception_full_emd_range --fb_split=False --PW=32 --PL=32 --cell_max_points=512 --cell_min_points=50 --encoder=inception --decoder=pointnet --stacked=False --rotation=True --pooling=mean --batch_size=128 --training=False --range_view=True

echo "-------------------------------"
# inception foldingnet
python main.py --loss_type=emd --save_path=${TOP_PATH}/inceptiongrid_full_emd_range --fb_split=False --PW=32 --PL=32 --cell_max_points=512 --cell_min_points=50 --encoder=inception --decoder=pointgrid --stacked=False --rotation=True --pooling=mean --batch_size=128 --training=False --range_view=True

