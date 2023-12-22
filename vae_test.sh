#!/bin/bash

python UBC_test.py\
    --exp=test_HGSC_UBC_vae\
    --dataset=UBC\
    --lr=1e-4\
    --img_size=224\
    --batch_size=16\
    --batch_size_test=1\
    --latent_img_size=28\
    --z_dim=256\
    --beta=1\
    --nb_channels=3\
    --model=UBC\
    --corr_type=corr_id\
    --params_id=100\
