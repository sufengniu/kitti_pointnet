#!/bin/bash

# top-down view 
TOP_PATH=/scratch4/sniu/pc

# pointnet
#python main.py --loss_type=emd --save_path=${TOP_PATH}/dgcnn_full_emd_l72 \
#               --fb_split=False --PW=32 --PL=32 --cell_max_points=512 \
#               --cell_min_points=50 --encoder=dgcnn --decoder=pointnet \
#               --stacked=False --rotation=False --pooling=max --training=True --latent_dim=72 --batch_size=256 --num_gpus=4

python main.py --loss_type=emd --save_path=${TOP_PATH}/dgcnn_full_emd_l36 \
               --fb_split=False --PW=32 --PL=32 --cell_max_points=512 \
               --cell_min_points=50 --encoder=dgcnn --decoder=pointnet \
               --stacked=False --rotation=False --pooling=max --training=True --latent_dim=36 --batch_size=256 --num_gpus=4

python main.py --loss_type=emd --save_path=${TOP_PATH}/dgcnn_full_emd_l9 \
               --fb_split=False --PW=32 --PL=32 --cell_max_points=512 \
               --cell_min_points=50 --encoder=dgcnn --decoder=pointnet \
               --stacked=False --rotation=False --pooling=max --training=True --latent_dim=9 --batch_size=256 --num_gpus=4


# pointnet
python main.py --loss_type=emd --save_path=${TOP_PATH}/dgcnn_full_emd_l72 \
               --fb_split=False --PW=32 --PL=32 --cell_max_points=512 \
               --cell_min_points=50 --encoder=dgcnn --decoder=pointnet \
               --stacked=False --rotation=False --pooling=max --training=False --latent_dim=72 --batch_size=256 --num_gpus=4

python main.py --loss_type=emd --save_path=${TOP_PATH}/dgcnn_full_emd_l36 \
               --fb_split=False --PW=32 --PL=32 --cell_max_points=512 \
               --cell_min_points=50 --encoder=dgcnn --decoder=pointnet \
               --stacked=False --rotation=False --pooling=max --training=False --latent_dim=36 --batch_size=256 --num_gpus=4

python main.py --loss_type=emd --save_path=${TOP_PATH}/dgcnn_full_emd_l9 \
               --fb_split=False --PW=32 --PL=32 --cell_max_points=512 \
               --cell_min_points=50 --encoder=dgcnn --decoder=pointnet \
               --stacked=False --rotation=False --pooling=max --training=False --latent_dim=9 --batch_size=256 --num_gpus=4

