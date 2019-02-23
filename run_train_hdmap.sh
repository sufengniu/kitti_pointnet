#!/bin/bash

# top-down view 
TOP_PATH=/scratch4/sniu/pc

# gin
python main.py --loss_type=emd --save_path=${TOP_PATH}/inception_full_emd_hd \
              --fb_split=False --PW=32 --PL=32 --cell_max_points=512 \
              --cell_min_points=50 --encoder=inception --decoder=pointnet \
              --stacked=False --rotation=True --pooling=mean --training=True \
              --latent_dim=18 --dataset=hdmap --batch_size=128



