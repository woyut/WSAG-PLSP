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
from PIL import Image

from PL_refinement.models.model_refine import ModelAGDsup as Model
from PL_refinement.dataset.data_refine import get_loader


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

    # define dataloader
    eval_data_loader = get_loader(
        batch_size=1,
        split_file=config["split_type"],
        data_dir=config["data_dir"],
        shuffle=False,
        obj_mask_dir_name=config["obj_mask_dir_name"],
        train=False,
        exo_obj_file=None, ego_obj_file=None, img_size=None,
    )
    
    logger.debug("Model:")
    logger.debug(model)
    model = torch.nn.DataParallel(model).cuda()
     
    model.eval()
    vall_num = 0
    vall_num_sum = 0
    
    os.makedirs(f"{config['data_dir']}/{config['split_type']}/trainset/{config['save_name']}", exist_ok=False)
    
    with torch.no_grad():
        for batch_data in tqdm(eval_data_loader):
            # get predicted logits
            aff_res = model(
                batch_data["input_image"], batch_data["part_feats"],
            )
            # remove padding
            pad_h, pad_w = batch_data["pads"][0]
            l = pad_w
            r = config["img_size"] - l
            t = pad_h
            b = config["img_size"] - t
            L = aff_res.shape[-1]
            l, r, t, b = l*L/config["img_size"], r*L/config["img_size"], t*L/config["img_size"], b*L/config["img_size"]
            l, r, t, b = int(l), int(r), int(t), int(b)
            pred = aff_res[:, :, t:b, l:r].detach()
            r_pred = F.interpolate(
                pred, 
                size=batch_data["whole_obj_mask"].shape[-2:],
                mode=INTERPOLATE_MODE,
            )
            # from logits to mask
            r_prob = F.sigmoid(r_pred.reshape(len(pred), -1))
            
            vall_num += 1
            vall_num_sum += len(pred)
            
            verbs = batch_data["verbs"]
            nouns = batch_data["nouns"]
            
            for bid in range(len(r_pred)):
                verb = verbs[bid]
                noun = nouns[bid]
                
                os.makedirs(f"{config['data_dir']}/{config['split_type']}/trainset/{config['save_name']}/{verb}/{noun}/", exist_ok=True)
                
                pp = r_prob[bid] > 0.95 # TODO: thresh
                # TODO: smooth?
                whole_obj_mask = batch_data["whole_obj_mask"][bid]
                pp = pp.reshape(*whole_obj_mask.shape).detach().cpu()
                # consider only the object region
                pp1 = whole_obj_mask * pp
                
                # save the refined mask
                assert not os.path.exists(f"{config['data_dir']}/{config['split_type']}/trainset/{config['save_name']}/{verb}/{noun}/{batch_data['input_path'][bid].split('/')[-1]}")
                pp_save = (pp1.numpy() * 255).astype(np.uint8)
                Image.fromarray(pp_save).save(f"{config['data_dir']}/{config['split_type']}/trainset/{config['save_name']}/{verb}/{noun}/{batch_data['input_path'][bid].split('/')[-1]}") 

if __name__ == "__main__":
    args = parse_args()
    config = load_config(args.config)
    main(config, )
