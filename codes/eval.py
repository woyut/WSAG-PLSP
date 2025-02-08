import sys
import os
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
from models.metric import KLD, SIM, NSS


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
    processed_heatmap = heatmap * 255 / np.max(heatmap)
    processed_heatmap = np.tile(processed_heatmap[:, :, np.newaxis], (1, 1, 3)).squeeze(2)
    processed_heatmap = processed_heatmap.astype('uint8')
    processed_heatmap = cv2.applyColorMap(processed_heatmap, cv2.COLORMAP_JET)
    # print(processed_heatmap.shape, image.shape)
    # assert processed_heatmap.shape == image.shape
    overlay = cv2.addWeighted(processed_heatmap, alpha, image, 1-alpha, 0) # TODO: [:, :, ::-1]
    cv2.imwrite(name, overlay) # TODO: , cv2.COLOR_BGR2RGB)
            

def main(config):
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

    logger.info(f'Set random seed to {1}, deterministic: '
                f'{config["deterministic"]}')
    INTERPOLATE_MODE = set_random_seed(1, deterministic=config["deterministic"])

    
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
        raise NotImplementedError
    
    num_parameters = sum([p.numel() for p in model.parameters()])
    logger.info(f'#Params: {num_parameters}')
    num_parameters = sum([p.numel() for p in model.encoder.parameters()])
    logger.info(f'#Encoder Params: {num_parameters}')
    num_parameters = sum([p.numel() for p in model.pred_decoder.parameters()])
    logger.info(f'#Final Decoder Params: {num_parameters}')
     
    eval_data_loader = get_loader(
        batch_size=1,
        img_size=config["img_size"], # follow LOCATE, Cross-View-AG, eval at 224*224
        split_file=config["split_type"],
        data_dir=config["data_dir"],
        shuffle=False,
        train=False,
        exo_obj_file=None, 
        ego_obj_file=None, 
        no_pad_gt=True
    )
    
    logger.debug("Model:")
    logger.debug(model)
    model = torch.nn.DataParallel(model).cuda()
     
    model.eval()
    vall_kld = 0.
    vall_sim = 0.
    vall_nss = 0.
    vall_num = 0
    vall_num_sum = 0
    
    with torch.no_grad():
        for batch_data in tqdm(eval_data_loader):
            aff_res, _, _ = model(
                batch_data["input_image"], batch_data["sent_feats"],
            )
            pred = aff_res.detach()
            
            r_pred = F.interpolate(
                pred, 
                size=batch_data["gt_mask"].shape[-2:],
                mode=INTERPOLATE_MODE,
            )
            
            gt_prob = batch_data["gt_mask_prob"].cuda().reshape(len(pred), -1)
            r_prob = F.softmax(r_pred.reshape(len(pred), -1), dim=1)
            
            kld_per_sample = KLD(r_prob, gt_prob, "none").sum(dim=1)
            kld = kld_per_sample.sum()
            sim = SIM(r_prob, gt_prob) * len(pred)
            nss = NSS(r_prob, gt_prob) * len(pred)
            vall_kld += kld
            vall_sim += sim
            vall_nss += nss
            vall_num += 1
            vall_num_sum += len(pred)
            
            verbs = batch_data["verbs"]
            nouns = batch_data["nouns"]
            
            
            for bid in range(len(r_pred)):
                pp = r_prob[bid] / r_prob[bid].max()
                
                ii = F.interpolate(
                    batch_data["input_image"][bid:bid+1], 
                    size=batch_data["gt_mask"].shape[-2:],
                    mode=INTERPOLATE_MODE,
                ).reshape(3, *batch_data["gt_mask"].shape[-2:])
                
                # plot_annotation(eval_data_loader.dataset.get_input_image_back(ii), 
                #                 batch_data["gt_mask"][bid].cpu().numpy().transpose(1, 2, 0), 
                #                 name=config['work_dir']+f'/img/AGD{vall_num}_{bid}_{kld_per_sample[bid]}_{verbs[bid]}.png')
                plot_annotation(eval_data_loader.dataset.get_input_image_back(ii), 
                                pp.reshape(*batch_data["gt_mask"][bid].shape).detach().cpu().numpy().transpose(1, 2, 0), 
                                name=config['work_dir']+f'/img/AGD{vall_num}_{bid}_{kld_per_sample[bid]}_{verbs[bid]}pred.png')
                
            
    logger.info(
        f"Result on AGD: \nKLD={vall_kld/vall_num_sum}, SIM={vall_sim/vall_num_sum}, NSS={vall_nss/vall_num_sum}")
    
if __name__ == "__main__":
    args = parse_args()
    config = load_config(args.config)
    main(config)
