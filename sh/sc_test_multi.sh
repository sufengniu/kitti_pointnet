TOP_PATH=/homes/schen/pct/share_with_Niu/kitti_pointnet/experiments
checkpoint_path=${TOP_PATH}/rv_pointnet_dim_18
dimension=18

train_batch_size=2048
learning_rate=0.002
num_epochs=281
save_freq=20

# Training
srun --gres gpu:1 -X  -c 3 --pty python /homes/schen/pct/share_with_Niu/kitti_pointnet/main.py  \
        --loss_type=emd --save_path=${checkpoint_path} \
        		--data_in_dir  /homes/schen/pct/kitti_pointnet/data/kitti/  \
               --fb_split=False --PW=32,16 --PL=32,16 --cell_max_points=512,512 \
               --cell_min_points=50,50 --encoder=pointnet --decoder=pointnet \
               --stacked=True --rotation=False --pooling=max --training=True  \
               --num_epochs=${num_epochs} --save_freq=${save_freq} \
              --latent_dim=${dimension} --dataset=kitti --batch_size=${train_batch_size} \
              --learning_rate ${learning_rate}  --num_gpus 1

# # top-down view 
# TOP_PATH=/homes/schen/pct/kitti_pointnet/experiments/kitti/range_view/ratio_distortion/pointnet
# evaluation_checkpoint_path=${TOP_PATH}/dim_18
# dimension=18


# # Evaluation
# test_batch_size=1024
# weights="model-280.ckpt"

# srun --gres gpu:1 -X  -c 3 --pty python /homes/schen/pct/share_with_Niu/kitti_pointnet/main.py  \
# 				        --loss_type=emd --save_path=${evaluation_checkpoint_path} \
# 				  --data_in_dir  /homes/schen/pct/kitti_pointnet/data/kitti/  \
#                --fb_split=False --PW=8 --PL=64 --cell_max_points=512 \
#                --cell_min_points=50 --encoder=pointnet --decoder=pointnet \
#                --stacked=False --rotation=False --pooling=max --training=False \
#               --latent_dim=${dimension} --dataset=kitti --batch_size=${test_batch_size} \
#               --weights ${weights} --num_gpus 1 --partition_mode range --load_prestored_data True