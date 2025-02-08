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
                 num_exo=1, PL_mode=None, aug4imgRatio=0., num_exo_pool=10):
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
        self.PL_mode = PL_mode
        self.aug4imgRatio = aug4imgRatio
        
        self.noun2verbs = {}
        self.noun2irrelevant_nouns = {}
        
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
                    if noun not in self.noun2verbs:
                        self.noun2verbs[noun] = []
                    self.noun2verbs[noun].append(verb)
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
                    if noun not in self.noun2verbs:
                        self.noun2verbs[noun] = []
                    self.noun2verbs[noun].append(verb)
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
        
        for noun in self.noun2verbs:
            self.noun2irrelevant_nouns[noun] = []
            for x in self.noun2verbs:
                flag = True
                for v in self.noun2verbs[noun]:
                    if v in self.noun2verbs[x]:
                        flag = False
                        break
                if flag:
                    self.noun2irrelevant_nouns[noun].append(x)
            assert len(self.noun2irrelevant_nouns[noun])

    def __len__(self):
        return len(self.input_paths)


    def __getitem__(self, idx):
        input_p = self.input_paths[idx]
        verb = self.input_verbs[idx]
        noun = self.input_nouns[idx]
        sent_feat = self.verbsFeat[verb]
        noun_feat = self.nounsFeat[noun]
        part_feat = self.partsFeat[verb+' '+noun]
        
        input_img_ori = Image.open(input_p)
        input_shape = [input_img_ori.size[1], input_img_ori.size[0]]
        input_img_noresize = self.transform_noresize(input_img_ori)
        input_img = F.interpolate(input_img_noresize.unsqueeze(0), size=256, mode='bilinear').squeeze(0)
        
        ego_boxes, ego_scores = self.ego_obj_dict[verb][noun][input_p.split('/')[-1]]
        ego_box_id = np.random.randint(0, len(ego_boxes))
        ego_box = ego_boxes[ego_box_id]

        all_exo_imgs = []
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
            exo_img = F.interpolate(exo_img_noresize.unsqueeze(0), size=224, mode='bilinear').squeeze(0)
            

            if self.exo_obj_dict is None or exo_p.split('/')[-1] not in self.exo_obj_dict[verb][noun] :
                assert False
            else:
                exo_boxes, exo_scores = self.exo_obj_dict[verb][noun][exo_p.split('/')[-1]]
                exo_box = exo_boxes[exo_box_id]
            exo_objbox_mask = self.box2mask(exo_box, exo_shape, L=224)
            all_exo_imgs.append(exo_img)
            all_exo_objbox_masks.append(exo_objbox_mask)
        if self.num_exo > 0:
            all_exo_imgs = torch.stack(all_exo_imgs, dim=0)
            all_exo_objbox_masks = torch.stack(all_exo_objbox_masks, dim=0)
        else:
            all_exo_imgs = torch.zeros(0, 3, self.img_size, self.img_size)
            all_exo_objbox_masks = torch.zeros(0, 1, self.img_size, self.img_size)
        

        if self.PL_mode == "refined":
            if os.path.exists(input_p.replace("egocentric", "PL_refined")):
                pl = torch.tensor(np.array(Image.open(
                    input_p.replace("egocentric", "PL_refined")
                ))).float() / 255.
            else:
                pl = torch.tensor(np.array(Image.open(
                    input_p.replace("egocentric", "PL").replace(".jpg", "_pl.png")
                ))).float() / 255.
        elif self.PL_mode is None:
            pl = torch.tensor(np.array(Image.open(
                input_p.replace("egocentric", "PL").replace(".jpg", "_pl.png")
            ))).float() / 255.
        else:
            raise NotImplementedError
        
        k_ratio = 3.0
        h, w = pl.shape
        pl = (pl >= 0.5)
        pl = pl.float().numpy()
        # Compute kernel size of the Gaussian filter. The kernel size must be odd.
        k_size = int(np.sqrt(h * w) / k_ratio)
        if k_size % 2 == 0:
            k_size += 1
        # Compute the heatmap using the Gaussian filter.
        pl = cv2.GaussianBlur(pl, (k_size, k_size), 0)
        pl = torch.tensor(pl).reshape(1, -1)
        gt_ori = pl.reshape(1, *input_shape)
        gt = F.interpolate(
            gt_ori.unsqueeze(0),
            size=256,
            mode="bilinear",
        ).squeeze(0)
        input_img, gt, rand_crop_origin = self.crop(input_img, gt, )

            
        if np.random.rand() < 0.5:
            input_img = torch.flip(input_img, dims=[2])
            gt = torch.flip(gt, dims=[2])
            
        exo_objbox_mask_patch = all_exo_objbox_masks.resize(self.num_exo, 14, 16, 14, 16).permute(0, 1, 3, 2, 4).reshape(self.num_exo, 14, 14, 256).mean(dim=-1)
        exo_objbox_mask_patch = (exo_objbox_mask_patch > 0).float().reshape(self.num_exo, 1, 196) # need this?
        
        if gt.max() == 0: # AGD20K/Seen/testset/GT/hold/bottle/bottle_000341.png
            gt = torch.ones_like(gt)
        gt_mask = gt / gt.max()
        gt_mask_prob = gt / gt.sum()
        
        vid = self.verb2vid[verb]
        nid = self.noun2nid[noun]
        
        if self.aug4imgRatio > 0.:
            if np.random.rand() < self.aug4imgRatio:
                neg_imgs = []
                for i in range(3):
                    if self.split == "Seen":
                        neg_n = np.random.choice(self.noun2irrelevant_nouns[noun])
                        neg_v = np.random.choice(self.noun2verbs[neg_n])
                    else:
                        neg_v = np.random.choice([v for v in os.listdir('/'.join(input_p.split('/')[:-3])) if v != verb])
                        neg_n = np.random.choice(os.listdir('/'.join(input_p.split('/')[:-3]) + '/' + neg_v))
                    neg_p = np.random.choice(os.listdir('/'.join(input_p.split('/')[:-3]) + '/' + neg_v + '/' + neg_n))
                    neg_img = Image.open('/'.join(input_p.split('/')[:-3]) + '/' + neg_v + '/' + neg_n + '/' + neg_p)
                    neg_img = self.transform_noresize(neg_img)
                    neg_img = F.interpolate(neg_img.unsqueeze(0), size=self.img_size//2, mode="bilinear")[0]
                    neg_imgs.append(neg_img)
                input_img1 = F.interpolate(input_img.unsqueeze(0), size=self.img_size//2, mode="bilinear")[0]
                gt_mask1 = F.interpolate(gt_mask.unsqueeze(0), size=self.img_size//2, mode="bilinear")[0]
                rand_aug_id = np.random.randint(0, 4)
                input_img = torch.zeros_like(input_img)
                gt_mask = torch.zeros_like(gt_mask)
                seq_len = (self.img_size//16)**2
                if rand_aug_id == 0:
                    input_img[:, :self.img_size//2, :self.img_size//2] = input_img1
                    gt_mask[:, :self.img_size//2, :self.img_size//2] = gt_mask1
                    input_img[:, :self.img_size//2, self.img_size//2:] = neg_imgs[0]
                    input_img[:, self.img_size//2:, :self.img_size//2] = neg_imgs[1]
                    input_img[:, self.img_size//2:, self.img_size//2:] = neg_imgs[2]
                elif rand_aug_id == 1:
                    input_img[:, :self.img_size//2, self.img_size//2:] = input_img1
                    gt_mask[:, :self.img_size//2, self.img_size//2:] = gt_mask1
                    input_img[:, :self.img_size//2, :self.img_size//2] = neg_imgs[0]
                    input_img[:, self.img_size//2:, :self.img_size//2] = neg_imgs[1]
                    input_img[:, self.img_size//2:, self.img_size//2:] = neg_imgs[2]
                elif rand_aug_id == 2:
                    input_img[:, self.img_size//2:, :self.img_size//2] = input_img1
                    gt_mask[:, self.img_size//2:, :self.img_size//2] = gt_mask1
                    input_img[:, :self.img_size//2, self.img_size//2:] = neg_imgs[0]
                    input_img[:, :self.img_size//2, :self.img_size//2] = neg_imgs[1]
                    input_img[:, self.img_size//2:, self.img_size//2:] = neg_imgs[2]
                elif rand_aug_id == 3:
                    input_img[:, self.img_size//2:, self.img_size//2:] = input_img1
                    gt_mask[:, self.img_size//2:, self.img_size//2:] = gt_mask1
                    input_img[:, :self.img_size//2, self.img_size//2:] = neg_imgs[0]
                    input_img[:, self.img_size//2:, :self.img_size//2] = neg_imgs[1]
                    input_img[:, :self.img_size//2, :self.img_size//2] = neg_imgs[2]
                gt_mask_prob = gt_mask / gt_mask.sum()
        
        return input_img, all_exo_imgs, gt_mask, gt_mask_prob, input_shape, sent_feat, noun_feat, part_feat, \
            verb, noun, exo_objbox_mask_patch, input_p, vid, nid

    
    def get_input_image_back(self, x):
        x = (x * self.pixel_std) + self.pixel_mean
        x = torch.clamp(x, min=0., max=1.)
        x = (x*255).permute(1, 2, 0).cpu().numpy().astype(np.uint8)
        return x
    
    def box2mask(self, box, input_shape, L=256):
        mask = torch.zeros(L, L)
        l, t, r, b = box.clone()
        l, t, r, b = np.floor(l/input_shape[1]*L), np.floor(t/input_shape[0]*L), np.ceil(r/input_shape[1]*L), np.ceil(b/input_shape[0]*L)
        r = max(l+1, r)
        b = max(t+1, b)
        l, t, r, b = int(l), int(t), int(r), int(b)
        mask[t:b, l:r] = 1
        assert mask.sum() > 0
        mask = mask / mask.sum()
        return mask.unsqueeze(0)
        

def get_loader(batch_size, data_dir, img_size, split_file, exo_obj_file, ego_obj_file, train, num_exo=1, num_exo_pool=10, shuffle=True, num_workers=8, PL_mode=None, aug4imgRatio=0., no_pad_gt=False):
    if train:
        dataset = DatasetAGD_train(data_dir, split_file, img_size, exo_obj_file, ego_obj_file, num_exo=num_exo, num_exo_pool=num_exo_pool, PL_mode=PL_mode, aug4imgRatio=aug4imgRatio)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn_train, num_workers=num_workers)
    else:
        dataset = DatasetAGD_test(data_dir, split_file, img_size, no_pad_gt=no_pad_gt)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn_test, num_workers=num_workers)
    return dataloader


def collate_fn_train(batch):
    input_imgs, exo_imgs, pseudo_gt_masks, pseudo_gt_masks_prob, input_shapes, \
        sent_feats, noun_feats, part_feats, verbs, nouns, exo_objbox_masks_patch, \
            input_ps, vids, nids, = list(zip(*batch))
    input_imgs = torch.stack(input_imgs, dim=0) # B 3 H W
    exo_imgs = torch.cat(exo_imgs, dim=0) # B*N 3 H W
    pseudo_gt_masks =  torch.stack(pseudo_gt_masks, dim=0) # B 1 H W
    pseudo_gt_masks_prob =  torch.stack(pseudo_gt_masks_prob, dim=0) # B 1 H W
    exo_objbox_masks_patch = torch.cat(exo_objbox_masks_patch, dim=0)
    B = input_imgs.shape[0]
    N_E = exo_imgs.shape[0] // B
    return {
        "input_image": input_imgs,
        "exo_image": exo_imgs,
        "gt_mask": pseudo_gt_masks, 
        "gt_mask_prob": pseudo_gt_masks_prob, 
        "input_shape": input_shapes,
        "sent_feats": torch.stack(sent_feats, dim=0),
        "noun_feats": torch.stack(noun_feats, dim=0),
        "part_feats": torch.stack(part_feats, dim=0),
        "verbs": verbs,
        "nouns": nouns,
        "valid_input": torch.ones(B,),
        "exo_objbox_mask_patch": exo_objbox_masks_patch.reshape(B*N_E, 196, 1), 
        "input_paths": input_ps,
        "vids": torch.tensor(vids),
        "nids": torch.tensor(nids),
    }


class DatasetAGD_test(torch.utils.data.Dataset):
    def __init__(self, data_dir, split, img_size, no_pad_gt=False):
        self.input_paths = []
        self.gt_paths = []
        self.input_verbs = []
        self.input_nouns = []
        self.verb2vid = {}
        self.vid2verb = []
        self.noun2nid = {}
        self.nid2noun = []
        
        self.no_pad_gt = no_pad_gt
        
        if split == "Seen":
            for iv, verb in enumerate(sorted(os.listdir(os.path.join(data_dir, split, "trainset", "egocentric")))):
                self.verb2vid[verb] = iv
                self.vid2verb.append(verb)
                for noun in sorted(os.listdir(os.path.join(data_dir, split, "trainset", "egocentric", verb))):       
                    if noun not in self.noun2nid:
                        self.noun2nid[noun] = len(self.noun2nid)
                        self.nid2noun.append(noun)
            for iv, verb in enumerate(os.listdir(os.path.join(data_dir, split, "testset", "egocentric"))):
                for noun in os.listdir(os.path.join(data_dir, split, "testset", "egocentric", verb)):
                    if os.path.exists(os.path.join(data_dir, split, "testset", "GT", verb, noun)):
                        for f in os.listdir(os.path.join(data_dir, split, "testset", "egocentric", verb, noun)):
                            if ".jpg" in f:
                                if os.path.exists(os.path.join(data_dir, split, "testset", "GT", verb, noun, f).replace('.jpg', '.png')):        
                                    self.input_paths.append(os.path.join(data_dir, split, "testset", "egocentric", verb, noun, f))
                                    self.gt_paths.append(os.path.join(data_dir, split, "testset", "GT", verb, noun, f).replace('.jpg', '.png'))
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
            for iv, verb in enumerate(os.listdir(os.path.join(data_dir, split, "testset", "egocentric"))):
                for noun in os.listdir(os.path.join(data_dir, split, "testset", "egocentric", verb)):
                    if os.path.exists(os.path.join(data_dir, split, "testset", "GT", verb, noun)):
                        for f in os.listdir(os.path.join(data_dir, split, "testset", "egocentric", verb, noun)):
                            if ".jpg" in f:
                                if os.path.exists(os.path.join(data_dir, split, "testset", "GT", verb, noun, f).replace('.jpg', '.png')):        
                                    self.input_paths.append(os.path.join(data_dir, split, "testset", "egocentric", verb, noun, f))
                                    self.gt_paths.append(os.path.join(data_dir, split, "testset", "GT", verb, noun, f).replace('.jpg', '.png'))
                                    self.input_verbs.append(verb)
                                    self.input_nouns.append(noun)
        else:
            raise NotImplementedError

        self.transform_noresize = transforms.Compose([
            _convert_image_to_rgb,
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])
        self.img_size = img_size
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
        sent_feat = self.verbsFeat[verb]
        noun_feat = self.nounsFeat[noun]
        part_feat = self.partsFeat[verb+' '+noun]
        
        input_img = Image.open(input_p)
        input_shape = [input_img.size[1], input_img.size[0]]
        input_img_no_resize = self.transform_noresize(input_img)
        
        if self.img_size:
            input_img = F.interpolate(
                input_img_no_resize.unsqueeze(0),
                size=self.img_size,
                mode="bilinear",
            ).squeeze(0)
        else:
            input_img = F.interpolate(
                input_img_no_resize.unsqueeze(0),
                size=224,
                mode="bilinear",
            ).squeeze(0)
        
        gt_ori = torch.tensor(np.array(Image.open(self.gt_paths[idx]))).float().reshape(1, *input_shape)
        assert self.no_pad_gt
        if self.img_size:
            gt = F.interpolate(
                gt_ori.unsqueeze(0),
                size=self.img_size,
                mode="bilinear",
            ).squeeze(0)
        else:
            gt = gt_ori
        
        if gt.max() == 0: # Seen/testset/GT/hold/bottle/bottle_000341.png
            gt = torch.ones_like(gt)
        gt_mask = gt / gt.max()
        gt_mask_prob = gt / gt.sum()
        
        return input_img, gt_mask, gt_mask_prob, input_shape, sent_feat, noun_feat, part_feat, verb, noun, input_p, vid, nid
    def get_input_image_back(self, x):
        x = (x * self.pixel_std) + self.pixel_mean
        x = torch.clamp(x, min=0., max=1.)
        x = (x*255).permute(1, 2, 0).cpu().numpy().astype(np.uint8)
        return x
        

def collate_fn_test(batch):
    input_imgs, pseudo_gt_masks, pseudo_gt_masks_prob, input_shapes, sent_feats, noun_feats, part_feats, \
        verbs, nouns, input_ps, vids, nids = list(zip(*batch))
    input_imgs = torch.stack(input_imgs, dim=0) # B 3 H W
    pseudo_gt_masks =  torch.stack(pseudo_gt_masks, dim=0) # B 1 H W
    pseudo_gt_masks_prob =  torch.stack(pseudo_gt_masks_prob, dim=0) # B 1 H W
    
    return {
        "input_image": input_imgs,
        "gt_mask": pseudo_gt_masks,
        "gt_mask_prob": pseudo_gt_masks_prob, 
        "input_shape": input_shapes,
        "sent_feats": torch.stack(sent_feats, dim=0),
        "noun_feats": torch.stack(noun_feats, dim=0),
        "part_feats": torch.stack(part_feats, dim=0), 
        "verbs": verbs,
        "nouns": nouns,
        "input_paths": input_ps,
        "vids": torch.tensor(vids),
        "nids": torch.tensor(nids),
    }
