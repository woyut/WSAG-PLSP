from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import torch
import os

import numpy as np
from torch.nn import functional as F
import torchvision.transforms.functional as TF
from torchvision.transforms import ToTensor, Normalize
from torchvision.transforms import InterpolationMode
BICUBIC = InterpolationMode.BICUBIC
import cv2


def input_class_for_seen(verb, noun):
    if verb == "throw" and noun == "javelin" or verb in ["pick_up", "drag", "hold"] and noun == "suitcase" or verb in ["carry", "pick_up", "hold"] and noun == "skis" or verb in ["hit", "hold", "swing"] and noun in ["tennis_racket", "badminton_racket", "baseball_bat"] or verb in ["carry", "hold"] and noun == "snowboard":
        return True
    return False

def input_class_for_unseen(verb, noun):
    if noun == "suitcase" or verb in ["hit", "hold", "swing"] and noun in ["tennis_racket", "badminton_racket", "baseball_bat"] or (noun == "wine_glass" and verb in ["hold", "pour"]) or verb == "hold" and noun in ["hammer", "scissors", "toothbrush", "fork"]:
        return True
    return False


def pad_to_square(tensor):
    _, h, w = tensor.shape
    max_dim = max(h, w)
    pad_height = max_dim - h
    pad_width = max_dim - w
    
    # Calculate padding for each side
    padding = (pad_width // 2, pad_width - pad_width // 2, pad_height // 2, pad_height - pad_height // 2)
    
    # Apply padding
    tensor_padded = F.pad(tensor, padding, mode='constant', value=0)
    
    return tensor_padded, pad_width // 2, pad_height // 2

def _convert_image_to_rgb(image):
    return image.convert("RGB")


class RandomCropBoth(object):
    def __init__(self, size):
        self.size = [size, size]

    def __call__(self, image, *masks):
        i, j, h, w = transforms.RandomCrop.get_params(image, output_size=self.size)
        image = TF.crop(image, i, j, h, w)
        ret_masks = []
        for mask in masks:
            mask = TF.crop(mask, i, j, h, w)
            ret_masks.append(mask)
        return image, *ret_masks, (i, j)


class DatasetAGD_train(torch.utils.data.Dataset):
    def __init__(self, data_dir, split, img_size, exo_obj_file, ego_obj_file, 
                 obj_mask_dir_name,
                 num_exo=1, num_exo_pool=10):
        self.input_paths = []
        self.gt_paths = []
        self.input_verbs = []
        self.input_nouns = []
        self.verb2vid = {}
        self.vid2verb = []
        self.noun2nid = {}
        self.nid2noun = []
        self.num_exo = num_exo
        self.num_exo_pool = num_exo_pool
        self.obj_mask_dir_name = obj_mask_dir_name
        
        if exo_obj_file is not None:
            self.exo_obj_dict = torch.load(exo_obj_file)
            self.ego_obj_dict = torch.load(ego_obj_file)
        else:
            self.exo_obj_dict = None
            self.ego_obj_dict = None
        
        
        if split == "Seen":
            for iv, verb in enumerate(sorted(os.listdir(os.path.join(data_dir, split, "trainset", "egocentric")))):
                self.verb2vid[verb] = iv
                self.vid2verb.append(verb)
                for noun in sorted(os.listdir(os.path.join(data_dir, split, "trainset", "egocentric", verb))):       
                    if noun not in self.noun2nid:
                        self.noun2nid[noun] = len(self.noun2nid)
                        self.nid2noun.append(noun)
            for iv, verb in enumerate(os.listdir(os.path.join(data_dir, split, "trainset", "egocentric"))):
                for noun in os.listdir(os.path.join(data_dir, split, "trainset", "egocentric", verb)):  
                    if not input_class_for_seen(verb, noun):
                        continue
                    for f in os.listdir(os.path.join(data_dir, split, "trainset", "egocentric", verb, noun)):
                        if ".jpg" in f:      
                            self.input_paths.append(os.path.join(data_dir, split, "trainset", "egocentric", verb, noun, f))
                            self.input_verbs.append(verb)
                            self.input_nouns.append(noun)
        elif split == "Unseen":
            # make sure the noun<->noun_id and verb<->verb_id mapping are consistent during training an testing
            for iv, verb in enumerate(sorted(os.listdir(os.path.join(data_dir, "Seen", "trainset", "egocentric")))):
                self.verb2vid[verb] = iv
                self.vid2verb.append(verb)
                for noun in sorted(os.listdir(os.path.join(data_dir, "Seen", "trainset", "egocentric", verb))):       
                    if noun not in self.noun2nid:
                        self.noun2nid[noun] = len(self.noun2nid)
                        self.nid2noun.append(noun)
            for iv, verb in enumerate(os.listdir(os.path.join(data_dir, "Unseen", "trainset", "egocentric"))):
                for noun in os.listdir(os.path.join(data_dir, "Unseen", "trainset", "egocentric", verb)):
                    if not input_class_for_unseen(verb, noun):
                        continue
                    
                    for f in os.listdir(os.path.join(data_dir, "Unseen", "trainset", "egocentric", verb, noun)):
                        if ".jpg" in f:
                            self.input_paths.append(os.path.join(data_dir, "Unseen", "trainset", "egocentric", verb, noun, f))
                            self.input_verbs.append(verb)
                            self.input_nouns.append(noun)
        else:
            raise NotImplementedError
        
        self.transform_noresize = transforms.Compose([
            _convert_image_to_rgb,
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])
        self.crop = RandomCropBoth(img_size)
        self.img_size = img_size
        self.pixel_mean=torch.Tensor([0.48145466, 0.4578275, 0.40821073]).view(-1, 1, 1) 
        self.pixel_std=torch.Tensor([0.26862954, 0.26130258, 0.27577711]).view(-1, 1, 1)
        
        
        self.nounsFeat = torch.load(os.path.join(data_dir, "sentenceFeatNounAGD.pth"))
        self.verbsFeat = torch.load(os.path.join(data_dir, "sentenceFeatVerbAGD.pth"))
        self.partsFeat = torch.load(os.path.join(data_dir, "sentenceFeatPartAGD.pth"))
        
        self.sims = {}
        for iv, verb in enumerate(os.listdir(os.path.join(data_dir, split, "trainset", "egocentric"))):
            self.sims[verb] = {}
            for noun in os.listdir(os.path.join(data_dir, split, "trainset", "egocentric", verb)):
                tmp = np.load(
                    os.path.join(data_dir, split, "trainset", "CLIPsim", f"{verb}_{noun}.npy"),
                    allow_pickle=True).item()
                self.sims[verb][noun] = tmp
        self.split = split    

    def __len__(self):
        return len(self.input_paths)


    def __getitem__(self, idx):
        input_p = self.input_paths[idx]
        verb = self.input_verbs[idx]
        noun = self.input_nouns[idx]
        sent_feat = self.verbsFeat[verb]
        noun_feat = self.nounsFeat[noun]
        part_feat = self.partsFeat[verb+' '+noun]
        
        # egocentric image: pad to square --- resize to 256 --- randcrop to 224
        input_img_ori = Image.open(input_p)
        input_shape = [input_img_ori.size[1], input_img_ori.size[0]]
        input_img_noresize = self.transform_noresize(input_img_ori)
        input_img_no_resize_padded, pad_w, pad_h = pad_to_square(input_img_noresize)
        input_shape_padded = input_img_no_resize_padded.shape[-1]
        input_img = F.interpolate(input_img_no_resize_padded.unsqueeze(0), size=256, mode='bilinear').squeeze(0)
        
        # egocentric object box
        ego_boxes, ego_scores = self.ego_obj_dict[verb][noun][input_p.split('/')[-1]]
        ego_box_id = np.random.randint(0, len(ego_boxes))
        ego_box = ego_boxes[ego_box_id]
        input_objbox_mask = self.box2mask(ego_box, pad_w, pad_h, input_shape_padded) # box mask, in padded image 256x256
        ego_obj_box_noresize = ego_box + np.array([pad_w, pad_h, pad_w, pad_h]) # box, in padded image original size
        
        # exocentric image, object box, object region, object mask in object region
        all_exo_imgs = []
        all_crop_weights = []
        all_exo_obj_regions = []
        all_exo_obj_region_obj_masks = []
        all_exo_objbox_masks = []
        sim_input_idx = self.sims[verb][noun]["image_dict"][input_p.split('/')[-1]+str(ego_box_id)]
        exo_pools = np.argsort(self.sims[verb][noun]['sim'][sim_input_idx])[-self.num_exo_pool:]
        for e in range(self.num_exo):
            exo_selected = np.random.choice(exo_pools)
            exo_p, exo_box_id = self.sims[verb][noun]["exo_dict_rev"][exo_selected]
            exo_p = os.path.join('/'.join(input_p.split('/')[:-1]).replace('egocentric', 'exocentric'), exo_p)
            # if not os.path.exists(exo_p):
            #     print(verb, noun, exo_p, exo_box_id, exo_selected)
            #     assert False
            exo_img_ori = Image.open(exo_p)
            exo_shape = [exo_img_ori.size[1], exo_img_ori.size[0]]
            exo_img_noresize = self.transform_noresize(exo_img_ori)
            exo_img_noresize_padded, pad_w_exo, pad_h_exo = pad_to_square(exo_img_noresize)
            exo_shape_padded = exo_img_noresize_padded.shape[-1]
            exo_img = F.interpolate(exo_img_noresize_padded.unsqueeze(0), size=256, mode='bilinear').squeeze(0)
            

            if self.exo_obj_dict is None or exo_p.split('/')[-1] not in self.exo_obj_dict[verb][noun] :
                assert False
            else:
                exo_boxes, exo_scores = self.exo_obj_dict[verb][noun][exo_p.split('/')[-1]]
                exo_box = exo_boxes[exo_box_id]
            exo_objbox_mask = self.box2mask(exo_box, pad_w_exo, pad_h_exo, exo_shape_padded) # padded img 256
            exo_whole_obj_mask_noresize = torch.tensor(np.array(Image.open(
                exo_p.replace("exocentric", self.obj_mask_dir_name).replace(".jpg", "_pl.png")
            ))).float() / 255. 
            exo_whole_obj_mask_noresize_padded, _, _ = pad_to_square(exo_whole_obj_mask_noresize.unsqueeze(0))
            
            exo_obj_box_noresize = exo_box + np.array([pad_w_exo, pad_h_exo, pad_w_exo, pad_h_exo]) # padded img ori
            l, t, r, b = [int(x) for x in exo_obj_box_noresize]
            exo_obj_region_noresize = exo_img_noresize_padded[:, t:b, l:r] # cropped padded img ori
            exo_obj_region_obj_mask_noresize = exo_whole_obj_mask_noresize_padded[:, t:b, l:r] # cropped padded img ori
            exo_obj_region_noresize_padded, _, _ = pad_to_square(exo_obj_region_noresize) # cropped padded img ori, padded again
            exo_obj_region_obj_mask_noresize_padded, _, _ = pad_to_square(exo_obj_region_obj_mask_noresize) # cropped padded img ori, padded again
            exo_obj_region = F.interpolate(
                exo_obj_region_noresize_padded.unsqueeze(0), size=self.img_size, mode="bilinear").squeeze(0) # cropped padded img ori, padded again 224
            exo_obj_region_obj_mask = F.interpolate(
                exo_obj_region_obj_mask_noresize_padded.unsqueeze(0), size=self.img_size, mode="bilinear").squeeze(0) # cropped padded img ori, padded again 224
            
            
            crop_weight = 1.
            if exo_whole_obj_mask_noresize_padded[:, t:b, l:r].sum() == 0:
                print("BAD EXO CROP:", exo_p, exo_box, exo_obj_box_noresize, exo_whole_obj_mask_noresize.shape)
                crop_weight = 0.
            
            exo_img, exo_objbox_mask, _ = self.crop(exo_img, exo_objbox_mask)
            
            all_exo_imgs.append(exo_img)
            all_crop_weights.append(crop_weight)
            all_exo_obj_regions.append(exo_obj_region)
            all_exo_obj_region_obj_masks.append(exo_obj_region_obj_mask)
            all_exo_objbox_masks.append(exo_objbox_mask)
        all_exo_imgs = torch.stack(all_exo_imgs, dim=0)
        all_crop_weights = torch.tensor(all_crop_weights)
        all_exo_obj_regions = torch.stack(all_exo_obj_regions, dim=0)
        all_exo_obj_region_obj_masks = torch.stack(all_exo_obj_region_obj_masks, dim=0)
        all_exo_objbox_masks = torch.stack(all_exo_objbox_masks, dim=0)
        

        # apply random crop to egocentric image & obj box
        input_img, input_objbox_mask, rand_crop_origin = self.crop(input_img, input_objbox_mask)
        rand_crop_origin_h, rand_crop_origin_w = rand_crop_origin
        rand_crop_box = [
            rand_crop_origin_w / 256 * input_shape_padded,
            rand_crop_origin_h / 256 * input_shape_padded,
            (rand_crop_origin_w+224) / 256 * input_shape_padded,
            (rand_crop_origin_h+224) / 256 * input_shape_padded,
        ]
        obj_box_in_rand_crop_box_noresize = np.array([
            max(ego_obj_box_noresize[0], rand_crop_box[0]),
            max(ego_obj_box_noresize[1], rand_crop_box[1]),
            min(ego_obj_box_noresize[2], rand_crop_box[2]),
            min(ego_obj_box_noresize[3], rand_crop_box[3]),
        ])
        obj_box_in_rand_crop_box = obj_box_in_rand_crop_box_noresize / input_shape_padded * 256
        obj_box_after_crop = obj_box_in_rand_crop_box - np.array([
            rand_crop_origin_w, rand_crop_origin_h, rand_crop_origin_w, rand_crop_origin_h])
        obj_box_after_crop = [int(x) for x in obj_box_after_crop]
        
        ego_obj_region_noresize = input_img_no_resize_padded[
            :, 
            int(obj_box_in_rand_crop_box_noresize[1]):int(obj_box_in_rand_crop_box_noresize[3]), 
            int(obj_box_in_rand_crop_box_noresize[0]):int(obj_box_in_rand_crop_box_noresize[2])
            ]
        
        # print(obj_box_in_rand_crop_box_noresize, obj_box_after_crop)
        if int(obj_box_in_rand_crop_box_noresize[3]) <= int(obj_box_in_rand_crop_box_noresize[1]) \
            or int(obj_box_in_rand_crop_box_noresize[2]) <= int(obj_box_in_rand_crop_box_noresize[0]) \
                or obj_box_after_crop[2] <= obj_box_after_crop[0] or obj_box_after_crop[3] <= obj_box_after_crop[1]:
            print("bad crop ego", input_p, input_objbox_mask.sum())
            all_crop_weights[:] = 0.
            ego_obj_region = torch.zeros(3, 224, 224)
            obj_box_after_crop = [0, 0, 224, 224]
        else:
            assert obj_box_after_crop[2] != obj_box_after_crop[0] and obj_box_after_crop[3] != obj_box_after_crop[1]
            ego_obj_region_noresize_padded, _, _ = pad_to_square(ego_obj_region_noresize)
            ego_obj_region = F.interpolate(ego_obj_region_noresize_padded.unsqueeze(0), size=self.img_size, mode="bilinear").squeeze(0)
            

        # apply random flip to egocentric image & obj box
        if np.random.rand() < 0.5:
            input_img = torch.flip(input_img, dims=[2])
            input_objbox_mask = torch.flip(input_objbox_mask, dims=[2])
            ego_obj_region = torch.flip(ego_obj_region, dims=[2])
            obj_box_after_crop[0], obj_box_after_crop[2] = 224 - obj_box_after_crop[2], 224 - obj_box_after_crop[0]
        
        return input_img, all_exo_imgs, input_shape, sent_feat, noun_feat, part_feat, \
            verb, noun, obj_box_after_crop, \
                ego_obj_region, all_exo_obj_regions, all_exo_obj_region_obj_masks, all_crop_weights

    
    def get_input_image_back(self, x):
        x = (x * self.pixel_std) + self.pixel_mean
        x = torch.clamp(x, min=0., max=1.)
        x = (x*255).permute(1, 2, 0).cpu().numpy().astype(np.uint8)
        return x
    
    def box2mask(self, box, pad_w, pad_h, input_shape_paded):
        L = 256
        mask = torch.zeros(L, L)
        l, t, r, b = box.clone()
        l += pad_w
        t += pad_h
        r += pad_w
        b += pad_h
        l, t, r, b = np.floor(l/input_shape_paded*L), np.floor(t/input_shape_paded*L), np.ceil(r/input_shape_paded*L), np.ceil(b/input_shape_paded*L)
        r = max(l+1, r)
        b = max(t+1, b)
        l, t, r, b = int(l), int(t), int(r), int(b)
        mask[t:b, l:r] = 1
        assert mask.sum() > 0
        mask = mask / mask.sum()
        return mask.unsqueeze(0)
        

def get_loader(batch_size, data_dir, img_size, split_file, exo_obj_file, ego_obj_file, train, obj_mask_dir_name, num_exo=1, num_exo_pool=10, shuffle=True, num_workers=8):
    if train:
        dataset = DatasetAGD_train(data_dir, split_file, img_size, exo_obj_file, ego_obj_file, obj_mask_dir_name=obj_mask_dir_name, num_exo=num_exo, num_exo_pool=num_exo_pool,)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn_train, num_workers=num_workers)
    else:
        dataset = DatasetAGD_test(data_dir, split_file, obj_mask_dir_name=obj_mask_dir_name)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn_test, num_workers=num_workers)
    return dataloader


def collate_fn_train(batch):
    input_imgs, exo_imgs, input_shapes, \
        sent_feats, noun_feats, part_feats, verbs, nouns, \
            ego_obj_boxs_after_crop, ego_obj_region, exo_obj_region, \
                exo_obj_region_obj_mask, crop_weights, \
                = list(zip(*batch))
    input_imgs = torch.stack(input_imgs, dim=0) # B 3 H W
    exo_imgs = torch.cat(exo_imgs, dim=0) # B*N 3 H W
    return {
        "input_image": input_imgs,
        "exo_image": exo_imgs,
        "input_shape": input_shapes,
        "sent_feats": torch.stack(sent_feats, dim=0),
        "noun_feats": torch.stack(noun_feats, dim=0),
        "part_feats": torch.stack(part_feats, dim=0),
        "verbs": verbs,
        "nouns": nouns,
        
        "input_obj_boxs": ego_obj_boxs_after_crop,
        "input_obj_region": torch.stack(ego_obj_region, dim=0),
        "exo_obj_region": torch.cat(exo_obj_region, dim=0),
        "crop_weights": torch.cat(crop_weights, dim=0),
        "exo_obj_region_obj_mask": torch.cat(exo_obj_region_obj_mask, dim=0)
    }



class DatasetAGD_test(torch.utils.data.Dataset):
    def __init__(self, data_dir, split, obj_mask_dir_name):
        self.input_paths = []
        self.gt_paths = []
        self.input_verbs = []
        self.input_nouns = []
        self.verb2vid = {}
        self.vid2verb = []
        self.noun2nid = {}
        self.nid2noun = []
        self.obj_mask_dir_name = obj_mask_dir_name
        
        if split == "Seen":
            for iv, verb in enumerate(sorted(os.listdir(os.path.join(data_dir, split, "trainset", "egocentric")))):
                self.verb2vid[verb] = iv
                self.vid2verb.append(verb)
                for noun in sorted(os.listdir(os.path.join(data_dir, split, "trainset", "egocentric", verb))):       
                    if noun not in self.noun2nid:
                        self.noun2nid[noun] = len(self.noun2nid)
                        self.nid2noun.append(noun)
            for iv, verb in enumerate(os.listdir(os.path.join(data_dir, split, "trainset", "egocentric"))):
                for noun in os.listdir(os.path.join(data_dir, split, "trainset", "egocentric", verb)):
                    if not input_class_for_seen(verb, noun):
                        continue
                    
                    for f in os.listdir(os.path.join(data_dir, split, "trainset", "egocentric", verb, noun)):
                        if ".jpg" in f:      
                            self.input_paths.append(os.path.join(data_dir, split, "trainset", "egocentric", verb, noun, f))
                            self.input_verbs.append(verb)
                            self.input_nouns.append(noun)
        elif split == "Unseen":
            for iv, verb in enumerate(sorted(os.listdir(os.path.join(data_dir, split, "trainset", "egocentric")))):
                self.verb2vid[verb] = iv
                self.vid2verb.append(verb)
                for noun in sorted(os.listdir(os.path.join(data_dir, split, "trainset", "egocentric", verb))):       
                    if noun not in self.noun2nid:
                        self.noun2nid[noun] = len(self.noun2nid)
                        self.nid2noun.append(noun)
            for iv, verb in enumerate(os.listdir(os.path.join(data_dir, split, "trainset", "egocentric"))):
                for noun in os.listdir(os.path.join(data_dir, split, "trainset", "egocentric", verb)):
                    if not input_class_for_unseen(verb, noun):
                        continue
                    for f in os.listdir(os.path.join(data_dir, split, "trainset", "egocentric", verb, noun)):
                        if ".jpg" in f:       
                            self.input_paths.append(os.path.join(data_dir, split, "trainset", "egocentric", verb, noun, f))
                            self.input_verbs.append(verb)
                            self.input_nouns.append(noun)
        else:
            raise NotImplementedError

        self.transform_noresize = transforms.Compose([
            _convert_image_to_rgb,
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])
        
        self.pixel_mean=torch.Tensor([0.48145466, 0.4578275, 0.40821073]).view(-1, 1, 1) 
        self.pixel_std=torch.Tensor([0.26862954, 0.26130258, 0.27577711]).view(-1, 1, 1)
        
        self.nounsFeat = torch.load(os.path.join(data_dir, "sentenceFeatNounAGD.pth"))
        self.verbsFeat = torch.load(os.path.join(data_dir, "sentenceFeatVerbAGD.pth"))
        self.partsFeat = torch.load(os.path.join(data_dir, "sentenceFeatPartAGD.pth"))

    def __len__(self):
        
        return len(self.input_paths)


    def __getitem__(self, idx):
        input_p = self.input_paths[idx]
        
        verb = self.input_verbs[idx]
        noun = self.input_nouns[idx]
        vid = self.verb2vid[verb]
        nid = self.noun2nid[noun]
        verb_feat = self.verbsFeat[verb]
        noun_feat = self.nounsFeat[noun]
        part_feat = self.partsFeat[verb+' '+noun]
        
        input_img = Image.open(input_p)
        input_shape = [input_img.size[1], input_img.size[0]]
        input_img_no_resize = self.transform_noresize(input_img)
        input_img_no_resize_padded, pad_w, pad_h = pad_to_square(input_img_no_resize)
        
        input_whole_obj_mask_noresize = torch.tensor(np.array(Image.open(
            input_p.replace("egocentric", self.obj_mask_dir_name).replace(".jpg", "_pl.png")
        ))).float() / 255.
        
        input_img = F.interpolate(
            input_img_no_resize_padded.unsqueeze(0),
            size=224,
            mode="bilinear",
        ).squeeze(0)
        pad_w = pad_w * 224 / input_img_no_resize_padded.shape[-1]
        pad_h = pad_h * 224 / input_img_no_resize_padded.shape[-1]
        
        
        return input_p, input_img, input_shape, verb_feat, noun_feat, part_feat, verb, noun, input_p, vid, nid, (pad_h, pad_w), input_whole_obj_mask_noresize
    def get_input_image_back(self, x):
        x = (x * self.pixel_std) + self.pixel_mean
        x = torch.clamp(x, min=0., max=1.)
        x = (x*255).permute(1, 2, 0).cpu().numpy().astype(np.uint8)
        return x

def collate_fn_test(batch):
    input_ps, input_imgs, input_shapes, verb_feats, noun_feats, part_feats,\
        verbs, nouns, input_ps, vids, nids, pads, whole_obj_masks = list(zip(*batch))
    input_imgs = torch.stack(input_imgs, dim=0) # B 3 H W
    
    return {
        "input_path": input_ps,
        "input_image": input_imgs,
        "input_shape": input_shapes,
        "verb_feats": torch.stack(verb_feats, dim=0),
        "noun_feats": torch.stack(noun_feats, dim=0),
        "part_feats": torch.stack(part_feats, dim=0),
        "verbs": verbs,
        "nouns": nouns,
        "vids": torch.tensor(vids),
        "nids": torch.tensor(nids),
        "pads": pads,
        "whole_obj_mask": torch.stack(whole_obj_masks, dim=0)
    }
    