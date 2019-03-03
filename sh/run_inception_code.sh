#!/bin/bash

# top-down view 
TOP_PATH=/scratch4/sniu/pc

# pointnet
# python main.py --loss_type=emd --save_path=${TOP_PATH}/inception_full_emd_l72 \
#                --fb_split=False --PW=32 --PL=32 --cell_max_points=512 \
#                --cell_min_points=50 --encoder=inception --decoder=pointnet \
#                --stacked=False --rotation=True --pooling=mean --training=True --latent_dim=72

# python main.py --loss_type=emd --save_path=${TOP_PATH}/inception_full_emd_l36 \
#                --fb_split=False --PW=32 --PL=32 --cell_max_points=512 \
#                --cell_min_points=50 --encoder=inception --decoder=pointnet \
#                --stacked=False --rotation=True --pooling=mean --training=True --latent_dim=36

# python main.py --loss_type=emd --save_path=${TOP_PATH}/inception_full_emd_l9 \
#                --fb_split=False --PW=32 --PL=32 --cell_max_points=512 \
#                --cell_min_points=50 --encoder=inception --decoder=pointnet \
#                --stacked=False --rotation=True --pooling=mean --training=True --latent_dim=9


# pointnet
#python main.py --loss_type=emd --save_path=${TOP_PATH}/inception_full_emd_l72 \
#               --fb_split=False --PW=32 --PL=32 --cell_max_points=512 \
#               --cell_min_points=50 --encoder=inception --decoder=pointnet \
#               --stacked=False --rotation=True --pooling=mean --training=False --latent_dim=72 --batch_size=128

#python main.py --loss_type=emd --save_path=${TOP_PATH}/inception_full_emd_l36 \
#               --fb_split=False --PW=32 --PL=32 --cell_max_points=512 \
#               --cell_min_points=50 --encoder=inception --decoder=pointnet \
#               --stacked=False --rotation=True --pooling=mean --training=False --latent_dim=36 --batch_size=128

python main.py --loss_type=emd --save_path=${TOP_PATH}/inception_full_emd_l9 \
               --fb_split=False --PW=32 --PL=32 --cell_max_points=512 \
               --cell_min_points=50 --encoder=inception --decoder=pointnet \
               --stacked=False --rotation=True --pooling=mean --training=False --latent_dim=9 --batch_size=128

