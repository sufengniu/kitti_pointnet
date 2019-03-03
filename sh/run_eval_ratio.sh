#!/bin/bash

TOP_PATH=/scratch4/sniu/pc

# inception
python main.py --loss_type=emd --save_path=${TOP_PATH}/inception_full_emd_10 --fb_split=False --PW=32 --PL=32 --cell_max_points=512 --cell_min_points=10 --encoder=inception --decoder=pointnet --stacked=False --rotation=True --pooling=mean --batch_size=128 --training=False --eval_all=False --weights=model-240.ckpt

# inception
python main.py --loss_type=emd --save_path=${TOP_PATH}/inception_full_emd_20 --fb_split=False --PW=32 --PL=32 --cell_max_points=512 --cell_min_points=20 --encoder=inception --decoder=pointnet --stacked=False --rotation=True --pooling=mean --batch_size=128 --training=False --eval_all=False

# inception
python main.py --loss_type=emd --save_path=${TOP_PATH}/inception_full_emd_30 --fb_split=False --PW=32 --PL=32 --cell_max_points=512 --cell_min_points=30 --encoder=inception --decoder=pointnet --stacked=False --rotation=True --pooling=mean --batch_size=128 --training=False --eval_all=False

# inception
python main.py --loss_type=emd --save_path=${TOP_PATH}/inception_full_emd_40 --fb_split=False --PW=32 --PL=32 --cell_max_points=512 --cell_min_points=40 --encoder=inception --decoder=pointnet --stacked=False --rotation=True --pooling=mean --batch_size=128 --training=False --eval_all=False
