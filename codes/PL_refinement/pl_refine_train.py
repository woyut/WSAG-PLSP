import sys
import os
main_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(main_dir)

import argparse
import yaml
import torch
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
import time
import logging
import random
import numpy as np
import cv2

from PL_refinement.models.model_refine import ModelAGDsup as Model
from PL_refinement.dataset.data_refine import get_loader as get_loader
from PL_refinement.dataset.data_refine import pad_to_square

from models.encoder_clip import VisionTransformer as CLIP

def set_random_seed(seed, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        INTERPOLATE_MODE = "nearest"
        torch.use_deterministic_algorithms(True)
    else:
        INTERPOLATE_MODE = "bilinear"
    return INTERPOLATE_MODE


def parse_args():
    parser = argparse.ArgumentParser(description='Finetuning on AGD20K')
    parser.add_argument('--config', type=str, help='Path to the configuration file', required=True)
    parser.add_argument('--seed', type=int, help='Seed', required=True)
    args = parser.parse_args()
    return args
 
  
def load_config(config_path):
    with open(config_path, 'r') as stream:
        config = yaml.safe_load(stream)
    return config


def plot_annotation(image, heatmap, alpha=0.5, name=""):
    """Plot the heatmap on the target image.

    Args:
        image: The target image.
        points: The annotated points.
        heatmap: The generated heatmap.
        alpha: The alpha value of the overlay image.
    """
    # Plot the overlay of heatmap on the target image.
    if heatmap is None:
        cv2.imwrite(name, cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    else:
        processed_heatmap = heatmap * 255 / np.max(heatmap)
        processed_heatmap = np.tile(processed_heatmap[:, :, np.newaxis], (1, 1, 3)).squeeze(2)
        processed_heatmap = processed_heatmap.astype('uint8')
        processed_heatmap = cv2.applyColorMap(processed_heatmap, cv2.COLORMAP_JET)
        # print(processed_heatmap.shape, image.shape)
        # assert processed_heatmap.shape == image.shape
        overlay = cv2.addWeighted(processed_heatmap, alpha, image, 1-alpha, 0) # TODO: [:, :, ::-1]
        cv2.imwrite(name, overlay) # TODO: , cv2.COLOR_BGR2RGB)
            

def main(config, seed):
    # set up logger
    os.makedirs(f"{config['work_dir']}", exist_ok=True)
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = f"{config['work_dir']}/{timestamp}.txt"
    logger = logging.getLogger("Train")
    file_handler = logging.FileHandler(log_file)
    console_handler = logging.StreamHandler(stream=sys.stdout)
    file_handler.setLevel("DEBUG")
    console_handler.setLevel("INFO")
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    logger.setLevel("DEBUG")
    
    logger.info(config)

    if not os.path.exists(f"{config['work_dir']}/ckpt"):
        os.makedirs(f"{config['work_dir']}/ckpt")
    if not os.path.exists(f"{config['work_dir']}/img"):
        os.makedirs(f"{config['work_dir']}/img")
    
    args_text = yaml.safe_dump(config, default_flow_style=False)
    logger.debug(f'======================Config:======================\n{args_text}')
    logger.info(f'Set random seed to {seed}, deterministic: '
                f'{config["deterministic"]}')
    INTERPOLATE_MODE = set_random_seed(seed, deterministic=config["deterministic"])

    # build model and load checkpoint
    model_config = config['model']
    model = Model(**model_config)
    
    load_config = config["load"]
    all_ckpt, encoder_ckpt = load_config["all_ckpt"], load_config["encoder_ckpt"]
    if all_ckpt:
        with open(all_ckpt, "rb") as f:
            state_dict = torch.load(f)["state_dict"]
        print("Loaded from ", all_ckpt)
        u, w = model.load_state_dict(state_dict, False)
        logger.debug(f'{u}, {w} are misaligned params in Model')
        for uu in u:
            logger.debug(uu)
        logger.debug("------------------------")
        for ww in w:
            logger.debug(ww)
    else:
        if config["model"]["encoder_type"] == "CLIP":
            state_dict = torch.jit.load(encoder_ckpt, map_location='cpu').float().state_dict()
            ckpt_dict = {}
            for k, v in state_dict.items():
                if "visual" in k:
                    ckpt_dict[k.split('visual.')[1]] = v
            u, w = model.encoder.load_state_dict(ckpt_dict, False)
            logger.debug(f'{u}, {w} are misaligned params in CLIP Encoder')
        else:
            assert encoder_ckpt is None
    
    num_parameters = sum([p.numel() for p in model.parameters()])
    logger.info(f'#Params: {num_parameters}')
    num_parameters = sum([p.numel() for p in model.encoder.parameters()])
    logger.info(f'#Encoder Params: {num_parameters}')
    num_parameters = sum([p.numel() for p in model.pred_decoder.parameters()])
    logger.info(f'#Final Decoder Params: {num_parameters}')    

    # define dataloader
    train_data_loader = get_loader(
        batch_size=config["batch_size"],
        img_size=config["img_size"],
        split_file=config["split_type"],
        data_dir=config["data_dir"],
        train=True,
        exo_obj_file=os.path.join(config["data_dir"], config["split_type"], "trainset", "det_wholeobj_exo.pth"), 
        ego_obj_file=os.path.join(config["data_dir"], config["split_type"], "trainset", "det_wholeobj_ego.pth"), 
        num_exo=config["num_exo"],
        obj_mask_dir_name=config["obj_mask_dir_name"]
    )

    
    
    # build optimizer
    encoder_params_id = list(map(id, model.encoder.parameters()))
    other_params = filter(
        lambda p: (id(p) not in encoder_params_id) and p.requires_grad==True, 
        model.parameters(),
    )
    encoder_params = filter(
        lambda p: p.requires_grad==True, 
        model.encoder.parameters(),
    )
    optimizer_config = config['optimizer']
    lr = optimizer_config['lr']
    lr_encoder_coeff = optimizer_config["lr_encoder_coeff"]
    all_params = [{'params': other_params}, 
                  {'params': encoder_params, 'lr': lr*lr_encoder_coeff}]
    
    num_epochs = optimizer_config["num_epochs"]
    save_epoch = optimizer_config["save_epoch"]
    accum_iter = optimizer_config["accum_iter"]
    betas = optimizer_config["betas"]
    wd = optimizer_config["wd"]
    optimizer = optim.AdamW(params=all_params, lr=lr, betas=betas, weight_decay=wd)

    if optimizer_config["sche_type"] == "step":
        lr_scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=optimizer_config["lr_step"], gamma=optimizer_config["lr_gamma"])
    elif optimizer_config["sche_type"] == "cos":
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=optimizer_config["max_iter"])

    logger.debug("Model:")
    logger.debug(model)
    logger.debug("Optimizer:")
    logger.debug(optimizer)
    # logger.debug("Scheduler:")
    # logger.debug(lr_scheduler)
    # logger.debug("Parameters:")
    # for key, value in model.named_parameters():
    #     logger.debug(f'{key},  {value.requires_grad}')
    
    model = torch.nn.DataParallel(model).cuda()
    loss_config = config["loss"]
    total_iter = 0
    
    best_align_loss = 10000.
    
    frozen_feature = CLIP()
    state_dict = torch.jit.load(encoder_ckpt, map_location='cpu').float().state_dict()
    ckpt_dict = {}
    for k, v in state_dict.items():
        if "visual" in k:
            ckpt_dict[k.split('visual.')[1]] = v
    u, w = frozen_feature.load_state_dict(ckpt_dict, False)
    frozen_feature = frozen_feature.cuda().eval()
    
    def get_frozen_feat(x):
        return frozen_feature(x)[1]
    
    for epoch in range(num_epochs):
        if total_iter >= optimizer_config["max_iter"]:
            break
        model.train()
        all_CLIP_align_loss = 0.0
        all_num = 0
        acc_num = 0
        logger.info(f"============Training Epoch {epoch}============")
        for batch_data in tqdm(train_data_loader):
            if total_iter >= optimizer_config["max_iter"]:
                break
            
            # get predicted logits: Bx1xLxL
            aff_res = model(
                batch_data["input_image"], # 224x224
                batch_data["part_feats"], 
            )
            L = aff_res.shape[-1]
            
            # get feature map of the exo obj region
            exo_CLIP_feat = get_frozen_feat(batch_data["exo_obj_region"].cuda()) # 224x224
            exo_CLIP_feat = exo_CLIP_feat[:, 1:].detach()
            B_e, _, C = exo_CLIP_feat.shape
            B = B_e // config["num_exo"]
            exo_CLIP_feat = exo_CLIP_feat.reshape(B_e, 14, 14, C).permute(0, 3, 1, 2) # B*num_exo x C x 14 x 14
            # get exo obj mask for the exo obj region
            exo_obj_region_obj_mask = batch_data["exo_obj_region_obj_mask"].cuda().reshape(B_e, 224, 224).reshape(B_e, 14, 16, 14, 16).permute(0, 1, 3, 2, 4).reshape(B_e, 1, 14, 14, 256).mean(dim=-1)
            exo_obj_region_obj_mask = exo_obj_region_obj_mask > 0.5
            # apply mask pooling
            exo_masked_patch_sum = (
                exo_CLIP_feat * exo_obj_region_obj_mask).sum(dim=[2, 3]) / (
                    exo_obj_region_obj_mask.sum(dim=[2, 3]) + 1e-8)
            
            # transform predicted logits to predicted mask: Bx1x224x224
            cur_size = L
            pred_mask = F.sigmoid(aff_res.reshape(B, -1))
            pred_mask = pred_mask.reshape(1, B, cur_size, cur_size)
            pred_mask = F.interpolate(pred_mask, size=config["img_size"], mode=INTERPOLATE_MODE).squeeze(0)
            pred_mask_bin = 1 - pred_mask
            if config["pred_mask_temperature"] > 1.:
                pred_mask_bin = torch.sigmoid(config["pred_mask_temperature"] * (pred_mask_bin - 0.5))
            
            # crop the predicted mask according to the ego obj region
            pred_mask_bin_crop = [] 
            for m, (l, t, r, b) in zip(pred_mask_bin, batch_data["input_obj_boxs"]):
                if r == l or b == t:
                    mm = F.interpolate(m.unsqueeze(0).unsqueeze(0), size=14, mode=INTERPOLATE_MODE).squeeze(0)
                else:
                    mm = F.interpolate(m[t:b, l:r].unsqueeze(0).unsqueeze(0), size=14, mode=INTERPOLATE_MODE).squeeze(0)
                mm, _, _ = pad_to_square(mm)
                pred_mask_bin_crop.append(mm)
            pred_mask_bin_crop = torch.stack(pred_mask_bin_crop, dim=0)
            # get feature map of the ego obj region
            input_CLIP_feat = get_frozen_feat(batch_data["input_obj_region"].cuda())
            input_CLIP_feat = input_CLIP_feat[:, 1:].detach()
            B, _, C = input_CLIP_feat.shape
            input_CLIP_feat = input_CLIP_feat.reshape(B, 14, 14, C).permute(0, 3, 1, 2)
            # apply mask pooling
            input_masked_patch_sum = (
                input_CLIP_feat * pred_mask_bin_crop).sum(dim=[2, 3]) / (
                    pred_mask_bin_crop.sum(dim=[2, 3]) + 1e-8)
            # compute sim loss
            input_masked_patch_sum_expand = input_masked_patch_sum.unsqueeze(1).expand(-1, config["num_exo"], -1).reshape(B_e, C)
            CLIP_align_loss = 1 - F.cosine_similarity(exo_masked_patch_sum.detach(), input_masked_patch_sum_expand, dim=1)
            CLIP_align_loss = (CLIP_align_loss *  batch_data["crop_weights"].cuda()).sum() / (batch_data["crop_weights"].cuda().sum() + 1e-8)
            
            
            cur_loss = loss_config["CLIP_align_coeff"] * CLIP_align_loss
            all_num += 1
            all_CLIP_align_loss += CLIP_align_loss.detach().item()
            
            cur_loss /= accum_iter
            cur_loss.backward()
            
            if all_num and all_num % accum_iter == 0:
                optimizer.step()
                optimizer.zero_grad()
                acc_num += 1
                if optimizer_config["sche_type"] == "cos":
                    lr_scheduler.step()
                total_iter += 1
            
            
            if all_num % (len(train_data_loader) // 20) == 0:
                pp = pred_mask[0]
                ii = F.interpolate(
                    batch_data["input_image"][0:1], 
                    size=224,
                    mode=INTERPOLATE_MODE,
                ).reshape(3, 224, 224)
                plot_annotation(train_data_loader.dataset.get_input_image_back(ii), 
                                pp.reshape(1, 224, 224).detach().cpu().numpy().transpose(1, 2, 0), 
                                name=config['work_dir']+f'/img/AGDtrain{all_num}pred.png')
                if batch_data["crop_weights"][0]:
                    plot_annotation(train_data_loader.dataset.get_input_image_back(batch_data["exo_obj_region"][0]), 
                                    None, 
                                    name=config['work_dir']+f'/img/AGDtrain{all_num}exocrop.png')
                    plot_annotation(train_data_loader.dataset.get_input_image_back(batch_data["exo_obj_region"][0] * batch_data["exo_obj_region_obj_mask"][0]), 
                                    None, 
                                    name=config['work_dir']+f'/img/AGDtrain{all_num}exocropmasked.png')
                    plot_annotation(train_data_loader.dataset.get_input_image_back(batch_data["exo_image"][0]), 
                                    None, 
                                    name=config['work_dir']+f'/img/AGDtrain{all_num}exo.png')
                    input_obj_crop_masked = batch_data["input_obj_region"].cuda() * F.interpolate(pred_mask_bin_crop, size=224, mode=INTERPOLATE_MODE)
                    plot_annotation(train_data_loader.dataset.get_input_image_back(input_obj_crop_masked[0].detach().cpu()), 
                                    None, 
                                    name=config['work_dir']+f'/img/AGDtrain{all_num}inputcrop_masked.png')
                    plot_annotation(train_data_loader.dataset.get_input_image_back(batch_data["input_obj_region"][0]), 
                                    None, 
                                    name=config['work_dir']+f'/img/AGDtrain{all_num}inputcrop.png')
            
           
        if optimizer_config["sche_type"] == "step":
            lr_scheduler.step()
        logger.info(
            f"CLIP align loss: {all_CLIP_align_loss / all_num}")
        logger.info(
            f"learning rate:{optimizer.state_dict()['param_groups'][0]['lr']}\n")
        
        if epoch % save_epoch == 0:
            torch.save({'optimizer': optimizer.state_dict(),
                        'state_dict': model.module.state_dict()}, os.path.join(config['work_dir'], "ckpt", f'epoch{epoch}_aam.ckpt'))
        if all_CLIP_align_loss < best_align_loss:
            best_align_loss = all_CLIP_align_loss
            torch.save({'optimizer': optimizer.state_dict(),
                        'state_dict': model.module.state_dict()}, os.path.join(config['work_dir'], "ckpt", f'best.ckpt'))
        

if __name__ == "__main__":
    args = parse_args()
    config = load_config(args.config)
    main(config, args.seed)
