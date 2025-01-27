#!/bin/bash

python vae_train.py\
    --exp=test_MC_UBC_vae_2nd_edition\
    --dataset=UBC\
    --category=wood\
    --lr=1e-4\
    --num_epochs=100\
    --img_size=224\
    --batch_size=32\
    --batch_size_test=16\
    --latent_img_size=14\
    --z_dim=192\
    --beta=1\
    --nb_channels=3\
    --model=UBC\
    --corr_type=corr_id\
    --force_train\
    --intest

