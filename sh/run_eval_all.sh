#!/bin/bash

# top-down view 
TOP_PATH=/scratch4/sniu/pc

# echo "-------------- pointnet -----------------"
# # pointnet
# python main.py --loss_type=emd --save_path=/scratch4/sniu/pc/pointnet_full_emd --fb_split=False --PW=32 --PL=32 --cell_max_points=512 --cell_min_points=50 --encoder=pointnet --decoder=pointnet --stacked=False --rotation=False --pooling=max --training=False --range_view=False

# echo "--------------- pointgrid ----------------"
# # pointnet foldingnet
# python main.py --loss_type=emd --save_path=/scratch4/sniu/pc/pointgrid_full_emd --fb_split=False --PW=32 --PL=32 --cell_max_points=512 --cell_min_points=50 --encoder=pointnet --decoder=pointgrid --stacked=False --rotation=False --pooling=max --training=False --range_view=False

# echo "--------------- dgcnn ----------------"
# # dgcnn 
# python main.py --loss_type=emd --save_path=${TOP_PATH}/dgcnn_full_emd --fb_split=False --PW=32 --PL=32 --cell_max_points=512 --cell_min_points=50 --encoder=dgcnn --decoder=pointnet --stacked=False --batch_size=128 --rotation=False --pooling=max --training=False --range_view=False

# echo "--------------- dgcnn grid----------------"
# # dgcnn foldingnet
# python main.py --loss_type=emd --save_path=${TOP_PATH}/dgcnngrid_full_emd --fb_split=False --PW=32 --PL=32 --cell_max_points=512 --cell_min_points=50 --encoder=dgcnn --decoder=pointgrid --stacked=False --batch_size=128 --rotation=False --pooling=max --training=False --range_view=False

# echo "--------------- inception ----------------"
# # inception base
# # python main.py --loss_type=emd --save_path=${TOP_PATH}/inception_base_full_emd --fb_split=False --PW=32 --PL=32 --cell_max_points=512 --cell_min_points=50 --encoder=inception --decoder=pointnet --stacked=False --rotation=False --pooling=max --batch_size=128 --training=False --range_view=False

# echo "---------------- inception grid ---------------"
# # inception base foldingnet
# # python main.py --loss_type=emd --save_path=${TOP_PATH}/inceptiongrid_base_full_emd --fb_split=False --PW=32 --PL=32 --cell_max_points=512 --cell_min_points=50 --encoder=inception --decoder=pointgrid --stacked=False --rotation=False --pooling=max --batch_size=128 --training=False --range_view=False


# echo "---------------- GIN ---------------"
# # inception
# python main.py --loss_type=emd --save_path=${TOP_PATH}/inception_full_emd --fb_split=False --PW=32 --PL=32 --cell_max_points=512 --cell_min_points=50 --encoder=inception --decoder=pointnet --stacked=False --rotation=True --pooling=mean --batch_size=128 --training=False --range_view=False

# echo "---------------- GIN grid ---------------"
# # inception foldingnet
# python main.py --loss_type=emd --save_path=${TOP_PATH}/inceptiongrid_full_emd --fb_split=False --PW=32 --PL=32 --cell_max_points=512 --cell_min_points=50 --encoder=inception --decoder=pointgrid --stacked=False --rotation=True --pooling=mean --batch_size=128 --training=False --range_view=False



# # range view 
# TOP_PATH=/scratch3/sniu/pc

# # pointnet
# python main.py --loss_type=emd --save_path=/scratch4/sniu/pc/pointnet_full_emd_range --fb_split=False --PW=32 --PL=32 --cell_max_points=512 --cell_min_points=50 --encoder=pointnet --decoder=pointnet --stacked=False --rotation=False --pooling=max --training=False --range_view=True

# echo "-------------------------------"
# # pointnet foldingnet
# python main.py --loss_type=emd --save_path=/scratch4/sniu/pc/pointgrid_full_emd_range --fb_split=False --PW=32 --PL=32 --cell_max_points=512 --cell_min_points=50 --encoder=pointnet --decoder=pointgrid --stacked=False --rotation=False --pooling=max --training=False --range_view=True

# echo "-------------------------------"
# # dgcnn 
# python main.py --loss_type=emd --save_path=${TOP_PATH}/dgcnn_full_emd_range --fb_split=False --PW=32 --PL=32 --cell_max_points=512 --cell_min_points=50 --encoder=dgcnn --decoder=pointnet --stacked=False --batch_size=128 --rotation=False --pooling=max --training=False --range_view=True

# echo "-------------------------------"
# # dgcnn foldingnet
# python main.py --loss_type=emd --save_path=${TOP_PATH}/dgcnngrid_full_emd_range --fb_split=False --PW=32 --PL=32 --cell_max_points=512 --cell_min_points=50 --encoder=dgcnn --decoder=pointgrid --stacked=False --batch_size=128 --rotation=False --pooling=max --training=False --range_view=True

# echo "-------------------------------"
# # inception
# python main.py --loss_type=emd --save_path=${TOP_PATH}/inception_full_emd_range --fb_split=False --PW=32 --PL=32 --cell_max_points=512 --cell_min_points=50 --encoder=inception --decoder=pointnet --stacked=False --rotation=True --pooling=mean --batch_size=128 --training=False --range_view=True

# echo "-------------------------------"
# # inception foldingnet
# python main.py --loss_type=emd --save_path=${TOP_PATH}/inceptiongrid_full_emd_range --fb_split=False --PW=32 --PL=32 --cell_max_points=512 --cell_min_points=50 --encoder=inception --decoder=pointgrid --stacked=False --rotation=True --pooling=mean --batch_size=128 --training=False --range_view=True

# ratio distortion random
echo "--------------- random ratio distortion top down view ----------------"
python main.py --loss_type=emd --save_path=/scratch4/sniu/pc/pointnet_full_emd_l72 --fb_split=False --PW=32 --PL=32 --cell_max_points=512 --cell_min_points=50 --encoder=pointnet --decoder=pointnet --stacked=False --rotation=False --pooling=max --training=False --range_view=False --mode=random --latent_dim=72
echo "-------------------------------"
python main.py --loss_type=emd --save_path=/scratch4/sniu/pc/pointnet_full_emd_l36 --fb_split=False --PW=32 --PL=32 --cell_max_points=512 --cell_min_points=50 --encoder=pointnet --decoder=pointnet --stacked=False --rotation=False --pooling=max --training=False --range_view=False --mode=random --latent_dim=36
echo "-------------------------------"
python main.py --loss_type=emd --save_path=/scratch4/sniu/pc/pointnet_full_emd --fb_split=False --PW=32 --PL=32 --cell_max_points=512 --cell_min_points=50 --encoder=pointnet --decoder=pointnet --stacked=False --rotation=False --pooling=max --training=False --range_view=False --mode=random --latent_dim=18
echo "-------------------------------"
python main.py --loss_type=emd --save_path=/scratch4/sniu/pc/pointnet_full_emd_l9 --fb_split=False --PW=32 --PL=32 --cell_max_points=512 --cell_min_points=50 --encoder=pointnet --decoder=pointnet --stacked=False --rotation=False --pooling=max --training=False --range_view=False --mode=random --latent_dim=9
# echo "-------------------------------"
# python main.py --loss_type=emd --save_path=/scratch4/sniu/pc/pointnet_full_emd_range --fb_split=False --PW=32 --PL=32 --cell_max_points=512 --cell_min_points=50 --encoder=pointnet --decoder=pointnet --stacked=False --rotation=False --pooling=max --training=False --range_view=False --mode=random


# # ratio distortion kmeans
# echo "-------------- kmeans ratio distortion top down view -----------------"
# python main.py --loss_type=emd --save_path=/scratch4/sniu/pc/pointnet_full_emd_l72 --fb_split=False --PW=32 --PL=32 --cell_max_points=512 --cell_min_points=50 --encoder=pointnet --decoder=pointnet --stacked=False --rotation=False --pooling=max --training=False --range_view=False --mode=kmeans --latent_dim=72
# echo "-------------------------------"
# python main.py --loss_type=emd --save_path=/scratch4/sniu/pc/pointnet_full_emd_l36 --fb_split=False --PW=32 --PL=32 --cell_max_points=512 --cell_min_points=50 --encoder=pointnet --decoder=pointnet --stacked=False --rotation=False --pooling=max --training=False --range_view=False --mode=kmeans --latent_dim=36
# echo "-------------------------------"
# python main.py --loss_type=emd --save_path=/scratch4/sniu/pc/pointnet_full_emd --fb_split=False --PW=32 --PL=32 --cell_max_points=512 --cell_min_points=50 --encoder=pointnet --decoder=pointnet --stacked=False --rotation=False --pooling=max --training=False --range_view=False --mode=kmeans --latent_dim=18
# echo "-------------------------------"
# python main.py --loss_type=emd --save_path=/scratch4/sniu/pc/pointnet_full_emd_l9 --fb_split=False --PW=32 --PL=32 --cell_max_points=512 --cell_min_points=50 --encoder=pointnet --decoder=pointnet --stacked=False --rotation=False --pooling=max --training=False --range_view=False --mode=kmeans --latent_dim=9
# echo "-------------------------------"
# python main.py --loss_type=emd --save_path=/scratch4/sniu/pc/pointnet_full_emd_range --fb_split=False --PW=32 --PL=32 --cell_max_points=512 --cell_min_points=50 --encoder=pointnet --decoder=pointnet --stacked=False --rotation=False --pooling=max --training=False --range_view=False --mode=kmeans



# # ratio distortion random range
# echo "-------------- random ratio distortion range view -----------------"
# python main.py --loss_type=emd --save_path=/scratch4/sniu/pc/pointnet_full_emd_range --fb_split=False --PW=32 --PL=32 --cell_max_points=512 --cell_min_points=50 --encoder=pointnet --decoder=pointnet --stacked=False --rotation=False --pooling=max --training=False --range_view=True --mode=random
# echo "-------------------------------"
# python main.py --loss_type=emd --save_path=/scratch4/sniu/pc/pointnet_full_emd_range --fb_split=False --PW=32 --PL=32 --cell_max_points=512 --cell_min_points=40 --encoder=pointnet --decoder=pointnet --stacked=False --rotation=False --pooling=max --training=False --range_view=True --mode=random
# echo "-------------------------------"
# python main.py --loss_type=emd --save_path=/scratch4/sniu/pc/pointnet_full_emd_range --fb_split=False --PW=32 --PL=32 --cell_max_points=512 --cell_min_points=30 --encoder=pointnet --decoder=pointnet --stacked=False --rotation=False --pooling=max --training=False --range_view=True --mode=random
# echo "-------------------------------"
# python main.py --loss_type=emd --save_path=/scratch4/sniu/pc/pointnet_full_emd_range --fb_split=False --PW=32 --PL=32 --cell_max_points=512 --cell_min_points=20 --encoder=pointnet --decoder=pointnet --stacked=False --rotation=False --pooling=max --training=False --range_view=True --mode=random
# echo "-------------------------------"
# python main.py --loss_type=emd --save_path=/scratch4/sniu/pc/pointnet_full_emd_range --fb_split=False --PW=32 --PL=32 --cell_max_points=512 --cell_min_points=10 --encoder=pointnet --decoder=pointnet --stacked=False --rotation=False --pooling=max --training=False --range_view=True --mode=random



# # ratio distortion kmeans range
# echo "------------- kmeans ratio distortion range view ------------------"
# python main.py --loss_type=emd --save_path=/scratch4/sniu/pc/pointnet_full_emd_range --fb_split=False --PW=32 --PL=32 --cell_max_points=512 --cell_min_points=50 --encoder=pointnet --decoder=pointnet --stacked=False --rotation=False --pooling=max --training=False --range_view=True --mode=kmeans
# echo "-------------------------------"
# python main.py --loss_type=emd --save_path=/scratch4/sniu/pc/pointnet_full_emd_range --fb_split=False --PW=32 --PL=32 --cell_max_points=512 --cell_min_points=40 --encoder=pointnet --decoder=pointnet --stacked=False --rotation=False --pooling=max --training=False --range_view=True --mode=kmeans
# echo "-------------------------------"
# python main.py --loss_type=emd --save_path=/scratch4/sniu/pc/pointnet_full_emd_range --fb_split=False --PW=32 --PL=32 --cell_max_points=512 --cell_min_points=30 --encoder=pointnet --decoder=pointnet --stacked=False --rotation=False --pooling=max --training=False --range_view=True --mode=kmeans
# echo "-------------------------------"
# python main.py --loss_type=emd --save_path=/scratch4/sniu/pc/pointnet_full_emd_range --fb_split=False --PW=32 --PL=32 --cell_max_points=512 --cell_min_points=20 --encoder=pointnet --decoder=pointnet --stacked=False --rotation=False --pooling=max --training=False --range_view=True --mode=kmeans
# echo "-------------------------------"
# python main.py --loss_type=emd --save_path=/scratch4/sniu/pc/pointnet_full_emd_range --fb_split=False --PW=32 --PL=32 --cell_max_points=512 --cell_min_points=10 --encoder=pointnet --decoder=pointnet --stacked=False --rotation=False --pooling=max --training=False --range_view=True --mode=kmeans



# # ratio distortion top down view
# echo "-------------------------------"
# # inception
# python main.py --loss_type=emd --save_path=${TOP_PATH}/inception_full_emd --fb_split=False --PW=32 --PL=32 --cell_max_points=512 --cell_min_points=50 --encoder=inception --decoder=pointnet --stacked=False --rotation=True --pooling=mean --batch_size=128 --training=False --range_view=False

# echo "-------------------------------"
# # inception
# python main.py --loss_type=emd --save_path=${TOP_PATH}/inception_full_emd --fb_split=False --PW=32 --PL=32 --cell_max_points=512 --cell_min_points=40 --encoder=inception --decoder=pointnet --stacked=False --rotation=True --pooling=mean --batch_size=128 --training=False --range_view=False

# echo "-------------------------------"
# # inception
# python main.py --loss_type=emd --save_path=${TOP_PATH}/inception_full_emd --fb_split=False --PW=32 --PL=32 --cell_max_points=512 --cell_min_points=30 --encoder=inception --decoder=pointnet --stacked=False --rotation=True --pooling=mean --batch_size=128 --training=False --range_view=False

# echo "-------------------------------"
# # inception
# python main.py --loss_type=emd --save_path=${TOP_PATH}/inception_full_emd --fb_split=False --PW=32 --PL=32 --cell_max_points=512 --cell_min_points=20 --encoder=inception --decoder=pointnet --stacked=False --rotation=True --pooling=mean --batch_size=128 --training=False --range_view=False

# echo "-------------------------------"
# # inception
# python main.py --loss_type=emd --save_path=${TOP_PATH}/inception_full_emd --fb_split=False --PW=32 --PL=32 --cell_max_points=512 --cell_min_points=10 --encoder=inception --decoder=pointnet --stacked=False --rotation=True --pooling=mean --batch_size=128 --training=False --range_view=False




# # ratio distortion range view
# echo "-------------------------------"
# # inception
# python main.py --loss_type=emd --save_path=${TOP_PATH}/inception_full_emd_range --fb_split=False --PW=32 --PL=32 --cell_max_points=512 --cell_min_points=50 --encoder=inception --decoder=pointnet --stacked=False --rotation=True --pooling=mean --batch_size=128 --training=False --range_view=True

# echo "-------------------------------"
# # inception
# python main.py --loss_type=emd --save_path=${TOP_PATH}/inception_full_emd_range --fb_split=False --PW=32 --PL=32 --cell_max_points=512 --cell_min_points=40 --encoder=inception --decoder=pointnet --stacked=False --rotation=True --pooling=mean --batch_size=128 --training=False --range_view=True

# echo "-------------------------------"
# # inception
# python main.py --loss_type=emd --save_path=${TOP_PATH}/inception_full_emd_range --fb_split=False --PW=32 --PL=32 --cell_max_points=512 --cell_min_points=30 --encoder=inception --decoder=pointnet --stacked=False --rotation=True --pooling=mean --batch_size=128 --training=False --range_view=True

# echo "-------------------------------"
# # inception
# python main.py --loss_type=emd --save_path=${TOP_PATH}/inception_full_emd_range --fb_split=False --PW=32 --PL=32 --cell_max_points=512 --cell_min_points=20 --encoder=inception --decoder=pointnet --stacked=False --rotation=True --pooling=mean --batch_size=128 --training=False --range_view=True

# echo "-------------------------------"
# # inception
# python main.py --loss_type=emd --save_path=${TOP_PATH}/inception_full_emd_range --fb_split=False --PW=32 --PL=32 --cell_max_points=512 --cell_min_points=10 --encoder=inception --decoder=pointnet --stacked=False --rotation=True --pooling=mean --batch_size=128 --training=False --range_view=True
