#!/bin/bash

# top-down view compression
TOP_PATH=/scratch4/sniu/pc/pointnet_full_emd

python main.py --loss_type=emd --save_path=${TOP_PATH} \
               --fb_split=False --PW=32 --PL=32 --cell_max_points=512 \
               --cell_min_points=50 --encoder=pointnet --decoder=pointnet \
               --stacked=False --rotation=False --pooling=max --training=False --compress_all_kitti=True

# running pytorch global level training
python global_level_representation/train.py -f ${TOP_PATH}/code.h5  --cuda --max-epochs 20 --iterations 16 --batch-size 50
