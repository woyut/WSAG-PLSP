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

from models.full_model import ModelAGDsup as Model
from dataset.data import get_loader as get_loader
from models.metric import KLD, SIM, KL_loss, NSS
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
        PL_mode=config["PL_mode"],
        aug4imgRatio=config["aug4imgRatio"]
    )
    val_data_loader = get_loader(
        batch_size=config["batch_size"],
        img_size=config["img_size"],
        split_file=config["split_type"],
        data_dir=config["data_dir"],
        shuffle=False,
        train=False,
        exo_obj_file=None, 
        ego_obj_file=None, 
        no_pad_gt=True,
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
    accum_iter = optimizer_config["accum_iter"]
    betas = optimizer_config["betas"]
    wd = optimizer_config["wd"]
    optimizer = optim.AdamW(params=all_params, lr=lr, betas=betas, weight_decay=wd)

    if optimizer_config["sche_type"] == "step":
        lr_scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=optimizer_config["lr_step"], gamma=optimizer_config["lr_gamma"])
    elif optimizer_config["sche_type"] == "cos":
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=optimizer_config["max_iter"], eta_min=1e-6)

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
    best_kld = 10000.
    best_sim = -10000.
    
    frozen_feature = CLIP()
    state_dict = torch.jit.load(encoder_ckpt, map_location='cpu').float().state_dict()
    ckpt_dict = {}
    for k, v in state_dict.items():
        if "visual" in k:
            ckpt_dict[k.split('visual.')[1]] = v
    u, w = frozen_feature.load_state_dict(ckpt_dict, False)
    frozen_feature = frozen_feature.cuda().eval()
    
    
    for epoch in range(num_epochs):
        if total_iter >= optimizer_config["max_iter"]:
            break
        model.train()
        all_loss = 0.0
        all_kl_loss = 0.
        all_sim_loss = 0.
        all_cls_loss = 0.0
        all_noun_sim_loss = 0.
        all_part_sim_loss = 0.
        all_num = 0
        acc_num = 0
        logger.info(f"============Training Epoch {epoch}============")
        for batch_data in tqdm(train_data_loader):
            if total_iter >= optimizer_config["max_iter"]:
                break
            if len(batch_data["input_image"]) == 1:
                continue # may cause cuda Bug
            
            if config["num_exo"] > 0:
                aff_res, sim_loss, exo_cls_res, pred_noun, pred_part = model(
                    batch_data["input_image"], batch_data["sent_feats"], 
                    batch_data["exo_image"], 
                    batch_data["exo_objbox_mask_patch"], num_exo=config["num_exo"],
                )
            else:
                aff_res, pred_noun, pred_part = model(
                    batch_data["input_image"], batch_data["sent_feats"], 
                )
            
            noun_sim_loss = (1 - F.cosine_similarity(pred_noun, batch_data["noun_feats"].cuda(), dim=2)).mean()
            part_sim_loss = (1 - F.cosine_similarity(pred_part, batch_data["part_feats"].cuda(), dim=2)).mean()
            
            if not (aff_res.shape[2] == batch_data["gt_mask"].shape[2] and aff_res.shape[3] == batch_data["gt_mask"].shape[3]):
                r_pred = F.interpolate(
                    aff_res, 
                    size=batch_data["gt_mask"].shape[-2:],
                    mode=INTERPOLATE_MODE,
                )
            else:
                r_pred = aff_res
            kl_loss = KL_loss(r_pred, batch_data["gt_mask_prob"].cuda(), batch_data["valid_input"].cuda())
            
            vids = batch_data["vids"].long().cuda().unsqueeze(1).expand(-1, config["num_exo"],).reshape(-1)
            if config["num_exo"] > 0:
                exo_cls_loss = F.cross_entropy(exo_cls_res, vids, reduction='mean')
                sim_loss = sim_loss.mean()
            else:
                sim_loss = torch.zeros(1,).cuda()
                exo_cls_loss = torch.zeros(1,).cuda()
            
            r_prob = F.softmax(r_pred.reshape(len(r_pred), -1), dim=1)
            gt_prob = batch_data["gt_mask_prob"].reshape(len(r_pred), -1)
            
            cur_loss = loss_config["kl_loss_coeff"] * kl_loss + \
                loss_config["sim_loss_coeff"] * sim_loss + \
                    loss_config["exo_cls_coeff"] * exo_cls_loss + \
                        loss_config["noun_sim_coeff"] * noun_sim_loss + \
                            loss_config["part_sim_coeff"] * part_sim_loss
            all_num += 1
            all_loss += cur_loss.detach().item()
            all_kl_loss += kl_loss.detach().item()
            all_sim_loss += sim_loss.detach().item()
            all_cls_loss += exo_cls_loss.detach().item()
            all_noun_sim_loss += noun_sim_loss.detach().item()
            all_part_sim_loss += part_sim_loss.detach().item()
            
            cur_loss /= accum_iter
            cur_loss.backward()
            
            if all_num and all_num % accum_iter == 0:
                optimizer.step()
                optimizer.zero_grad()
                acc_num += 1
                if optimizer_config["sche_type"] == "cos":
                    lr_scheduler.step()
                total_iter += 1
            
            
        if optimizer_config["sche_type"] == "step":
            lr_scheduler.step()
        logger.info(
            f"Training loss: {all_loss / all_num}, KL loss: {all_kl_loss / all_num}, Sim loss: {all_sim_loss / all_num}, Exo CLS loss: {all_cls_loss / all_num}, \n"
            f"Noun sim loss: {all_noun_sim_loss / all_num}, Part sim loss: {all_part_sim_loss / all_num}")
        logger.info(
            f"learning rate:{optimizer.state_dict()['param_groups'][0]['lr']}\n")
        
        model.eval()
        vall_kld = 0.
        vall_sim = 0.
        vall_nss = 0.
        vall_num = 0
        vall_num_sum = 0
        
        vall_noun_sim = 0.
        vall_part_sim = 0.
        with torch.no_grad():
            for batch_data in tqdm(val_data_loader):
                aff_res, pred_noun, pred_part = model(
                    batch_data["input_image"], batch_data["sent_feats"],
                )
                pred = aff_res.detach()
                
                r_pred = F.interpolate(
                    pred, 
                    size=batch_data["gt_mask"].shape[-2:],
                    mode=INTERPOLATE_MODE,
                )
                
                noun_sim_loss = (1 - F.cosine_similarity(pred_noun, batch_data["noun_feats"].cuda(), dim=2)).mean()
                part_sim_loss = (1 - F.cosine_similarity(pred_part, batch_data["part_feats"].cuda(), dim=2)).mean()
                vall_noun_sim += noun_sim_loss.detach().item()
                vall_part_sim += part_sim_loss.detach().item()
                
                gt_prob = batch_data["gt_mask_prob"].cuda().reshape(len(pred), -1)
                r_prob = F.softmax(r_pred.reshape(len(pred), -1), dim=1)
                
                kld = KLD(r_prob, gt_prob) * len(pred)
                sim = SIM(r_prob, gt_prob) * len(pred)
                nss = NSS(r_prob, gt_prob) * len(pred)
                vall_kld += kld
                vall_sim += sim
                vall_nss += nss
                vall_num += 1
                vall_num_sum += len(pred)
                
        logger.info(
            f"Result on AGD: \nKLD={vall_kld/vall_num_sum}, SIM={vall_sim/vall_num_sum}, NSS={vall_nss/vall_num_sum}"
            f"\nnoun sim: {vall_noun_sim/vall_num}, part sim: {vall_part_sim/vall_num}")
        
        
        if vall_kld/vall_num_sum < best_kld:
            best_kld = vall_kld/vall_num_sum
            torch.save({'optimizer': optimizer.state_dict(),
                        'state_dict': model.module.state_dict()}, os.path.join(config['work_dir'], "ckpt", f'bestKLD.ckpt'))
            logger.info(f"New best KLD: {vall_kld/vall_num_sum}, {vall_sim/vall_num_sum}, {vall_nss/vall_num_sum}")
        # if vall_sim/vall_num_sum > best_sim:
        #     best_sim = vall_sim/vall_num_sum
        #     torch.save({'optimizer': optimizer.state_dict(),
        #                 'state_dict': model.module.state_dict()}, os.path.join(config['work_dir'], "ckpt", f'bestSIM.ckpt'))
        
        

if __name__ == "__main__":
    args = parse_args()
    config = load_config(args.config)
    main(config, args.seed)
