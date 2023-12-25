# các giá trị cần output
# identity của ảnh (local + global) AND/OR đường dẫn
# Tên ảnh 
# Giá trị loss (MAD)

import os
import time
import argparse
import numpy as np
from tqdm import tqdm
from skimage.metrics import structural_similarity
import torch
from vqvae import VQVAE
from scipy.ndimage import label
from scipy.ndimage.morphology import binary_dilation
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from PIL import Image
from utils import (get_train_dataloader,
                   get_test_dataloader,
                   load_model_parameters,
                   load_vqvae,
                   parse_args
                   )
import pandas as pd

def ssim(a, b, win_size):
    "Structural di-SIMilarity: SSIM"
    a = a.detach().cpu().permute(1, 2, 0).numpy()
    b = b.detach().cpu().permute(1, 2, 0).numpy()
    # print(a)
    # b = gaussian_filter(b, sigma=2)

    try:
        score, full = structural_similarity(a, b, #multichannel=True,
            channel_axis=2, full=True, win_size=win_size,data_range=1.0)
    except ValueError: # different version of scikit img
        score, full = structural_similarity(a, b, multichannel=True,
            channel_axis=2, full=True, win_size=win_size,data_range=1.0)
    #return 1 - score, np.median(1 - full, axis=2)  # Return disim = (1 - sim)
    return 1 - score, np.product((1 - full), axis=2)

def get_error_pixel_wise(model, x_loc, x_glo, loss="rec_loss"):
    
    
    x_rec, _ = model(x_loc, x_glo)
    
    return x_rec

def create_segmentation(amaps, amaps2, rec_maps, dataset):
    if isinstance(amaps, torch.Tensor):
        
        amaps = amaps.detach().cpu().numpy()
    torch.set_printoptions(threshold=10_000)
    
    mask = (amaps>= 0.01).astype(np.int8) #NOTE 0.59 if M=512 !!!
    print(np.amax(mask))
    
    mask2 = ((amaps2 >= 0.01)).astype(np.int8) #0.01 for VQVAE

    mask2 = binary_dilation(mask2, iterations=4).astype(np.uint8)

    ## get intersection
    L, _ = label(mask2)
    lbls_interest = L[(mask == 1)]
    lbls_interest = lbls_interest[lbls_interest != 0]
    mask_ = mask.copy()
    mask = (np.isin(L, lbls_interest)).astype(np.int8)
    print(np.amax(mask))
    return mask, mask2, mask_, L

def test(args):
    ''' livestock testing pipeline '''
    # device = what you will use on this function
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.force_cpu else "cpu"
    )
    print("Pytorch device:", device)

    seed = 0
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    np.random.seed(seed)

    checkpoints_dir ="./torch_checkpoints"
    if not os.path.isdir(checkpoints_dir):
        os.mkdir(checkpoints_dir)
    checkpoints_saved_dir ="./torch_checkpoints_saved"

    predictions_dir ="./" + args.dataset + "_predictions"
    if not os.path.isdir(predictions_dir):
        os.mkdir(predictions_dir)

    # Load dataset
    train_dataloader, train_dataset = get_train_dataloader(
        args,
        fake_dataset_size=256,
    )
    # NOTE force test batch size to be 1
    args.batch_size_test = 1
    # fake_dataset_size=None leads a test on all the test dataset
    test_dataloader, test_dataset = get_test_dataloader(
        args,
        fake_dataset_size=None
    )

    # Load model
    model = load_vqvae(args)
    # device: 1 concept của pytorch

    model.to(device)

    try:
        file_name = f"{args.exp}_{args.params_id}.pth"
        model = load_model_parameters(model, file_name, checkpoints_dir,
            checkpoints_saved_dir, device)
    except FileNotFoundError:
        raise RuntimeError("The model checkpoint does not exist !")

    dissimilarity_func = ssim

    classes = {}

    model.eval()

    rec_loss = []

    output_csv = {
        'local_img_name': [],
        'global_img_name': [],
        'local_img_path': [],
        'rec_loss': []
    }

    

    # t
    pbar = tqdm(test_dataloader)
    for imgs_loc, imgs_glo, _, imgs_loc_path, imgs_glo_name, imgs_loc_name in pbar:
        imgs_loc = imgs_loc.to(device)
        imgs_glo = imgs_glo.to(device)
        

        # if args.dataset in ["livestock","mvtec","miad"]:
        #     # gt is a segmentation mask
        #     gt_np = gt[0].permute(1, 2, 0).cpu().numpy()[..., 0]
        #     gt_np = (gt_np - np.amin(gt_np)) / (np.amax(gt_np) - np.amin(gt_np))

        
            

        with torch.no_grad():
            x_rec = get_error_pixel_wise(model, imgs_loc, imgs_glo)
            x_rec = model.mean_from_lambda(x_rec)

        if args.dataset == "UBC":
            score, ssim_map = dissimilarity_func(x_rec[0], imgs_loc[0], 11)

        # Structural Similarity Index Measure (SSIM)
        amaps = ((ssim_map - np.amin(ssim_map)) / (np.amax(ssim_map)
        - np.amin(ssim_map)))
    
         
        rec_loss.append(torch.median(torch.from_numpy(amaps)))     
        # bỏ data vào trong các trường của output_csv
        output_csv['local_img_name'].append(imgs_loc_name)
        output_csv["global_img_name"].append(imgs_glo_name)
        output_csv["local_img_path"] .append(imgs_loc_path)
        output_csv["rec_loss"].append(torch.median(torch.from_numpy(amaps)))


        print(float("{:.2f}".format(rec_loss[-1])))
        m_rec = np.mean(rec_loss)
        print(m_rec)
        pbar.set_description(f"mean Reconstruction loss: {m_rec:.3f}")

        ori = imgs_loc[0].permute(1, 2, 0).cpu().numpy()
        # gt = gt[0].permute(1, 2, 0).cpu().numpy()
        rec = x_rec[0].detach().permute(1, 2, 0).cpu().numpy()
        path_to_save = args.dataset + '_predictions/'
        img_to_save = Image.fromarray((ori * 255).astype(np.uint8))
        img_to_save.save(path_to_save + 'ori.png')
        # img_to_save = Image.fromarray((gt_np * 255).astype(np.uint8))
        # img_to_save.save(path_to_save + 'gt.png')
        img_to_save = Image.fromarray((rec * 255).astype(np.uint8))
        img_to_save.save(path_to_save + 'rec.png')
        cm = plt.get_cmap('jet')
        amaps = cm(amaps)
        img_to_save = Image.fromarray((amaps[..., :3] * 255).astype(np.uint8))
        img_to_save.save(path_to_save + 'final_amap.png')

    m_rec = np.mean(rec_loss)
    print("Mean rec loss on", args.category, args.defect, m_rec)

    # ra ngoài for loop thì in cái dataframe 
    dataframe = pd.DataFrame(output_csv)
    dataframe.to_csv('D:\source\GitHub\data\csv')

    return m_rec



def test_on_train(args, model):
    ''' livestock testing pipeline '''
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.force_cpu else "cpu"
    )
    print("Pytorch device:", device)

    seed = 0
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    np.random.seed(seed)

    checkpoints_dir ="./torch_checkpoints"
    if not os.path.isdir(checkpoints_dir):
        os.mkdir(checkpoints_dir)
    checkpoints_saved_dir ="./torch_checkpoints_saved"

    predictions_dir ="./" + args.dataset + "_predictions"
    if not os.path.isdir(predictions_dir):
        os.mkdir(predictions_dir)

    # Load dataset
    train_dataloader, train_dataset = get_train_dataloader(
        args,
        fake_dataset_size=256,
    )
    # NOTE force test batch size to be 1
    args.batch_size_test = 1
    # fake_dataset_size=None leads a test on all the test dataset
    test_dataloader, test_dataset = get_test_dataloader(
        args,
        fake_dataset_size=None
        
    )

    

    dissimilarity_func = ssim

    classes = {}

    model.eval()

    rec_loss = []

    pbar = tqdm(test_dataloader)
    for imgs_loc, imgs_glo, _ in pbar:
        imgs_loc = imgs_loc.to(device)
        imgs_glo = imgs_glo.to(device)
        # if args.dataset in ["livestock","mvtec","miad"]:
            # gt is a segmentation mask
            # gt_np = gt[0].permute(1, 2, 0).cpu().numpy()[..., 0]
            # gt_np = (gt_np - np.amin(gt_np)) / (np.amax(gt_np) - np.amin(gt_np))

        
        
        
        with torch.no_grad():
            x_rec = get_error_pixel_wise(model, imgs_loc, imgs_glo)
            x_rec = model.mean_from_lambda(x_rec)

        if args.dataset == "UBC":
            score, ssim_map = dissimilarity_func(x_rec[0], imgs_loc[0], 11)

        
        amaps = ((ssim_map - np.amin(ssim_map)) / (np.amax(ssim_map)
        - np.amin(ssim_map)))
        rec_loss.append(torch.mean(torch.from_numpy(amaps)))  
        m_rec_loss = np.mean(rec_loss)
        print(m_rec_loss)
        pbar.set_description(f"mean rec loss: {m_rec_loss:.3f}")
        # pbar.set_description(f"mean Reconstruction loss: {m_rec:.3f}")

        ori = imgs_loc[0].permute(1, 2, 0).cpu().numpy()
        # gt = gt[0].permute(1, 2, 0).cpu().numpy()
        rec = x_rec[0].detach().permute(1, 2, 0).cpu().numpy()
        path_to_save = args.dataset + '_predictions/'
        img_to_save = Image.fromarray((ori * 255).astype(np.uint8))
        img_to_save.save(path_to_save + 'ori.png')
        # img_to_save = Image.fromarray((gt_np * 255).astype(np.uint8))
        # img_to_save.save(path_to_save + 'gt.png')
        img_to_save = Image.fromarray((rec * 255).astype(np.uint8))
        img_to_save.save(path_to_save + 'rec.png')
        cm = plt.get_cmap('jet')
        amaps = cm(amaps)
        img_to_save = Image.fromarray((amaps[..., :3] * 255).astype(np.uint8))
        img_to_save.save(path_to_save + 'final_amap.png')


        mad = torch.mean(torch.abs(model.mu - torch.mean(model.mu,
            dim=(0,1))), dim=(0,1))

        mad = mad.detach().cpu().numpy()

        mad = ((mad - np.amin(mad)) / (np.amax(mad)
            - np.amin(mad)))

        mad = mad.repeat(8, axis=0).repeat(8, axis=1)

        # MAD metric
        amaps = mad

    m_rec_loss = np.mean(rec_loss)
    print("Mean rec loss on", args.category, args.defect, m_rec_loss)

    

    return m_rec_loss

if __name__ == "__main__":
    args = parse_args()

    if args.dataset == "UBC":
        m_auc = test(
            args,
            )
