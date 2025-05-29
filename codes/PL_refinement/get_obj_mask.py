import argparse
import os
import copy

import numpy as np
import json
import torch
from PIL import Image, ImageDraw, ImageFont

import cv2
import matplotlib.pyplot as plt
import detectron2.data.transforms as T
from vlpart.vlpart import build_vlpart
from segment_anything import build_sam, SamPredictor
from segment_anything.utils.amg import remove_small_regions

from tqdm import tqdm




if __name__ == "__main__":
    parser = argparse.ArgumentParser("Segment-Anything-and-Name-It Demo", add_help=True)
    parser.add_argument(
        "--vlpart_checkpoint", type=str, default="swinbase_part_0a0000.pth", help="path to checkpoint file"
    )
    parser.add_argument(
        "--sam_checkpoint", type=str, default="sam_vit_h_4b8939.pth", help="path to checkpoint file"
    )
    parser.add_argument(
        "--split_name", type=str, default="Seen/trainset", help="the split of AGD20K"
    )
    parser.add_argument(
        "--view_name", type=str, default="egocentric", help="exo or ego"
    )
    parser.add_argument(
        "--AGD_dir", type=str, default="/path/to/AGD20K", help="the directory of AGD20K"
    )
    parser.add_argument(
        "--output_name", type=str, default="whole_obj_mask_ego", help="the name of the directory to save the generated masks"
    )
    parser.add_argument("--box_threshold", type=float, default=0.3, help="box threshold")
    parser.add_argument("--text_threshold", type=float, default=0.25, help="text threshold")

    parser.add_argument("--device", type=str, default="cuda", help="running on cuda")
    args = parser.parse_args()

    # cfg
    vlpart_checkpoint = args.vlpart_checkpoint
    sam_checkpoint = args.sam_checkpoint

    box_threshold = args.box_threshold
    text_threshold = args.box_threshold
    device = args.device
    
    

    # initialize VLPart
    vlpart = build_vlpart(checkpoint=vlpart_checkpoint)
    vlpart.to(device=device)

    # initialize SAM
    sam_predictor = SamPredictor(build_sam(checkpoint=sam_checkpoint).to(device=device))

    split = args.split_name
    output_name = args.output_name
    data_dir = args.AGD_dir
    view_name = args.view_name
    
    for verb in tqdm(os.listdir(f"{data_dir}/{split}/{view_name}")):
        for noun in os.listdir(os.path.join(f"{data_dir}/{split}/{view_name}", verb)):
            os.makedirs(os.path.join(f"{data_dir}/{split}/", output_name, verb, noun), exist_ok=True)
            for img_name in os.listdir(os.path.join(f"{data_dir}/{split}/{view_name}", verb, noun)):
                image_path = os.path.join(f"{data_dir}/{split}/{view_name}", verb, noun, img_name)
                
                # load image
                image = cv2.imread(image_path)
                original_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                # vlpart model inference
                preprocess = T.ResizeShortestEdge([800, 800], 1333)
                height, width = original_image.shape[:2]
                image = preprocess.get_transform(original_image).apply_image(original_image)
                image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
                inputs = {"image": image, "height": height, "width": width}
                with torch.no_grad():
                    predictions = vlpart.inference([inputs], text_prompt=noun.replace("_", " "))[0]

                boxes, masks = None, None
                filter_scores, filter_boxes, filter_classes = [], [], []

                if "instances" in predictions:
                    instances = predictions['instances'].to('cpu')
                    boxes = instances.pred_boxes.tensor if instances.has("pred_boxes") else None
                    scores = instances.scores if instances.has("scores") else None
                    classes = instances.pred_classes.tolist() if instances.has("pred_classes") else None

                    num_obj = len(scores)
                    max_score = -1.
                    for obj_ind in range(num_obj):
                        category_score = scores[obj_ind]
                        if category_score > 0.5:
                            filter_scores.append(category_score)
                            filter_boxes.append(boxes[obj_ind])
                            filter_classes.append(classes[obj_ind])
                            max_score = max(max_score, category_score)
                    if len(filter_boxes) == 0:
                        for obj_ind in range(num_obj):
                            category_score = scores[obj_ind]
                            if category_score > max_score:
                                filter_scores = [category_score,]
                                filter_boxes = [boxes[obj_ind],]
                                filter_classes = [classes[obj_ind],]
                                max_score = max(max_score, category_score)

                if len(filter_boxes) > 0:
                    # sam model inference
                    sam_predictor.set_image(original_image)

                    boxes_filter = torch.stack(filter_boxes)
                    transformed_boxes = sam_predictor.transform.apply_boxes_torch(boxes_filter, original_image.shape[:2])
                    masks, _, _ = sam_predictor.predict_torch(
                        point_coords=None,
                        point_labels=None,
                        boxes=transformed_boxes.to(device),
                        multimask_output=False,
                    )

                    # remove small disconnected regions and holes
                    fine_masks = []
                    flag = False
                    for im, mask in enumerate(masks.to('cpu').numpy()):  # masks: [num_masks, 1, h, w]
                        m = mask[0]
                        b_l, b_t, b_r, b_b = boxes_filter[im]
                        b_l, b_t, b_r, b_b = int(b_l), int(b_t), int(b_r), int(b_b)
                        boundary_sum = m[b_t, b_l:b_r].sum() + m[b_b-1, b_l:b_r].sum() + m[b_t:b_b, b_l].sum() + m[b_t:b_b, b_r-1].sum()
                        # Determine whether "1" or "0" corresponds to the foreground based on boundary sum
                        if boundary_sum > b_r-b_l+b_b-b_t:
                            flag = True
                            m[b_t:b_b, b_l:b_r] = 1 - m[b_t:b_b, b_l:b_r]
                        fine_masks.append(remove_small_regions(m, 400, mode="holes")[0])
                    masks = np.stack(fine_masks, axis=0)[:, np.newaxis]
                    masks = torch.from_numpy(masks).float()
                    mask = masks.max(dim=0, keepdim=True).values
                    mask = torch.nn.functional.interpolate(mask, size=[height, width], mode="bilinear", align_corners=False).squeeze()
                    
                    
                    # Normalize the float tensor values to range [0, 1]
                    mask = torch.clamp(mask, 0, 1)
                    # Convert the tensor to a numpy array and reshape if necessary
                    mask = mask.mul(255).byte().numpy()
                    # Convert the numpy array to a PIL Image
                    mask = Image.fromarray(mask, mode='L')  # 'L' mode for grayscale
                    # Save the PIL Image as a grayscale image
                    assert not os.path.exists(image_path.replace(view_name, output_name).replace(".jpg", "_pl.png"))
                    mask.save(image_path.replace(view_name, output_name).replace(".jpg", "_pl.png"))
    
    
    # # draw output image
    # plt.figure(figsize=(10, 10))
    # plt.imshow(original_image)

    # if len(filter_boxes) > 0:
    #     show_predictions_with_masks(filter_scores, filter_boxes, filter_classes,
    #                                 masks.to('cpu'), text_prompt)

    # plt.axis('off')
    # image_name = image_path.split('/')[-1]
    # plt.savefig(
    #     os.path.join(output_dir, "vlpart_sam_output_{}".format(image_name)),
    #     bbox_inches="tight", dpi=300, pad_inches=0.0
    # )
