#!/bin/bash

# top-down view 
TOP_PATH=/scratch4/sniu/pc

#pointnet
python plot.py --loss_type=emd --save_path=${TOP_PATH}/pointnet_full_emd \
               --fb_split=False --PW=32 --PL=32 --cell_max_points=512 \
               --cell_min_points=50 --encoder=pointnet --decoder=pointnet \
               --stacked=False --rotation=False --pooling=max --mode=random --dataset=hdmap

python plot.py --loss_type=emd --save_path=${TOP_PATH}/pointnet_full_emd \
              --fb_split=False --PW=32 --PL=32 --cell_max_points=512 \
              --cell_min_points=50 --encoder=pointnet --decoder=pointnet \
              --stacked=False --rotation=False --pooling=max --mode=kmeans --dataset=hdmap

python plot.py --loss_type=emd --save_path=${TOP_PATH}/pointnet_full_emd \
              --fb_split=False --PW=32 --PL=32 --cell_max_points=512 \
              --cell_min_points=50 --encoder=pointnet --decoder=pointnet \
              --stacked=False --rotation=False --pooling=max --mode=autoencoder --dataset=hdmap

python plot.py --loss_type=emd --save_path=${TOP_PATH}/inception_full_emd \
               --fb_split=False --PW=32 --PL=32 --cell_max_points=512 \
               --cell_min_points=50 --encoder=inception --decoder=pointnet \
               --stacked=False --rotation=True --pooling=mean --mode=autoencoder --batch_size=200 --dataset=hdmap
