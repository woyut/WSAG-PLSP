import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import os
from tqdm import tqdm

from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

sam_checkpoint = "sam_vit_h_4b8939.pth" # TODO
model_type = "vit_h"
device = "cuda"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
mask_generator = SamAutomaticMaskGenerator(
    model=sam,
    points_per_side=8,
    # pred_iou_thresh=0.86,
    # stability_score_thresh=0.92,
    # crop_n_layers=1,
    # crop_n_points_downscale_factor=2,
    min_mask_region_area=100,  # Requires open-cv to run post-processing
)
save_name = "SAM_masks_pps8"
split = "Seen" # TODO
data_dir = "/path/to/AGD20K" # TODO

exo_obj_dict = torch.load(f"{data_dir}/{split}/trainset/det_wholeobj_exo.pth")
ego_obj_dict = torch.load(f"{data_dir}/{split}/trainset/det_wholeobj_ego.pth")


for verb in os.listdir(f"{data_dir}/{split}/trainset/egocentric"):
    print(verb)
    for noun in tqdm(os.listdir(f"{data_dir}/{split}/trainset/egocentric/{verb}")):
        os.makedirs(f"{data_dir}/{split}/trainset/{save_name}/{verb}/{noun}", exist_ok=True)
        for p in os.listdir(f"{data_dir}/{split}/trainset/egocentric/{verb}/{noun}"):
            # print(p)
            image_p = f"{data_dir}/{split}/trainset/egocentric/{verb}/{noun}/{p}"
            image = cv2.imread(image_p)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            ego_boxes, ego_scores = ego_obj_dict[verb][noun][p]
            for ie, ego_box in enumerate(ego_boxes):
                l,t,r,b = [int(x) for x in ego_box]
                ego_crop = image[t:b, l:r]
                # print(ego_box, image.shape)

                masks = mask_generator.generate(ego_crop)
                for iim, mask in enumerate(masks):
                    ppp = image_p.replace("egocentric", save_name).replace(".jpg", f"_{ie}-{iim}.png")
                    assert not os.path.exists(ppp)
                    cv2.imwrite(ppp, mask['segmentation']*255)
            