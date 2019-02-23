#!/bin/bash

# top-down view 
TOP_PATH=/scratch4/sniu/pc


echo "--------------- inception ----------------"
# inception base
python main.py --loss_type=emd --save_path=${TOP_PATH}/inception_base_full_emd --fb_split=False --PW=32 --PL=32 --cell_max_points=512 --cell_min_points=50 --encoder=inception --decoder=pointnet --stacked=False --rotation=False --pooling=max --batch_size=128 --training=False --range_view=False

echo "---------------- inception grid ---------------"
# inception base foldingnet
python main.py --loss_type=emd --save_path=${TOP_PATH}/inceptiongrid_base_full_emd --fb_split=False --PW=32 --PL=32 --cell_max_points=512 --cell_min_points=50 --encoder=inception --decoder=pointgrid --stacked=False --rotation=False --pooling=max --batch_size=128 --training=False --range_view=False


echo "--------------- inception ----------------"
# inception base
python main.py --loss_type=emd --save_path=${TOP_PATH}/inception_base_mean_full_emd --fb_split=False --PW=32 --PL=32 --cell_max_points=512 --cell_min_points=50 --encoder=inception --decoder=pointnet --stacked=False --rotation=False --pooling=mean --batch_size=128 --training=False --range_view=False

echo "---------------- inception grid ---------------"
# inception base foldingnet
python main.py --loss_type=emd --save_path=${TOP_PATH}/inceptiongrid_base_mean_full_emd --fb_split=False --PW=32 --PL=32 --cell_max_points=512 --cell_min_points=50 --encoder=inception --decoder=pointgrid --stacked=False --rotation=False --pooling=mean --batch_size=128 --training=False --range_view=False
