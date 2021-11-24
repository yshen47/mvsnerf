import torch
from torch.utils.data import Dataset
import json
import numpy as np
import os
from PIL import Image
from torchvision import transforms as T
import ast
from .ray_utils import *


class BlenderDataset(Dataset):
    def __init__(self, args, split='train', load_ref=False):
        self.args = args
        self.scene = args.scene
        self.root_dir = args.datadir
        self.split = split
        downsample = args.imgScale_train if split=='train' else args.imgScale_test
        assert int(800*downsample)%32 == 0, \
            f'image width is {int(800*downsample)}, it should be divisible by 32, you may need to modify the imgScale'
        self.img_wh = (int(800*downsample), int(800*downsample))
        self.define_transforms()
        version_folders = []
        if os.path.exists(f'runs_fine_tuning/{self.args.expname}'):
            version_folders = sorted(
                [int(folder.split('_')[-1]) for folder in os.listdir(f'runs_fine_tuning/{self.args.expname}') if
                 'version' in folder], reverse=True)
        if len(version_folders) != 0:
            self.version_num = version_folders[0]
        else:
            self.version_num = ""
        self.project_root_path = f'runs_fine_tuning/{self.args.expname}/version_{self.version_num}'
        self.blender2opencv = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        if not load_ref:
            self.read_meta()

        self.white_back = True

    def read_meta(self):
        if self.split == 'all':
            with open(os.path.join(self.root_dir, f"transforms_train.json"), 'r') as f:
                train_meta = json.load(f)

            with open(os.path.join(self.root_dir, f"transforms_val.json"), 'r') as f:
                val_meta = json.load(f)
        else:
            with open(os.path.join(self.root_dir, f"transforms_{self.split}.json"), 'r') as f:
                self.meta = json.load(f)

        with open(os.path.join(f"/data/phi_theta_sample/{self.scene}", f"transforms_train.json"), 'r') as f:
            new_view_meta = json.load(f)

        if self.version_num != '':
            next_view_indices_pth = os.path.join(self.project_root_path, 'next_view_indices.txt')
            if os.path.exists(next_view_indices_pth):
                with open(next_view_indices_pth, 'r') as f:
                    lines = f.readlines()
                next_view_indices = lines[-1].strip().split(",")

        # sub select training views from pairing file
        if os.path.exists('configs/pairs.th'):
            name = os.path.basename(self.root_dir)
            if self.split != 'all':
                self.img_idx = torch.load('configs/pairs.th')[f'{name}_{self.split}']
                self.meta['frames'] = [self.meta['frames'][idx] for idx in self.img_idx]
                print(f'===> {self.split}ing index: {self.img_idx}')
            else:
                train_img_idx = torch.load('configs/pairs.th')[f'{name}_train']
                train_meta['frames'] = [train_meta['frames'][idx] for idx in train_img_idx]

                val_img_idx = torch.load('configs/pairs.th')[f'{name}_val']
                val_meta['frames'] = [val_meta['frames'][idx] for idx in val_img_idx]
                print(f'===> validating index: {val_img_idx}')
                assert train_meta["camera_angle_x"] == val_meta["camera_angle_x"]
                self.meta = train_meta
                self.meta['frames'] += val_meta['frames']

        if self.version_num != '':
            new_frames = {}
            for f in new_view_meta['frames']:
                file_path = f['file_path'].split('/')[-1]
                f['file_path'] = "/".join(f['file_path'].split('/')[-2:])
                new_frames[file_path] = f

            for next_view_index in next_view_indices:
                self.meta['frames'].append(new_frames[next_view_index])
            self.next_view_indices = next_view_indices
            if self.split != 'val':
                print(f'===> new view index: {next_view_indices}')
        else:
            self.next_view_indices = []
        w, h = self.img_wh
        self.focal = 0.5 * 800 / np.tan(0.5 * self.meta['camera_angle_x'])  # original focal length
        self.focal *= self.img_wh[0] / 800  # modify focal length to match size self.img_wh

        # bounds, common for all scenes
        self.near = 2.0
        self.far = 6.0
        self.bounds = np.array([self.near, self.far])

        # ray directions for all pixels, same for all images (same H, W, focal)
        self.directions = get_ray_directions(h, w, [self.focal,self.focal])  # (h, w, 3)


        self.image_paths = []
        self.poses = []
        self.all_rays = []
        self.all_rgbs = []
        self.all_masks = []
        self.item_types = [] # 0 for train, 1 for val, 2 for new
        for frame in self.meta['frames']:
            file_path = frame['file_path']
            if 'train' in file_path:
                self.item_types.append('train')
            elif 'val' in file_path:
                self.item_types.append('val')
            elif 'new' in file_path:
                self.item_types.append('new')
            else:
                raise ValueError
            pose = np.array(frame['transform_matrix']) @ self.blender2opencv
            self.poses += [pose]
            c2w = torch.FloatTensor(pose)

            image_path = os.path.join(self.root_dir + "/" + self.item_types[-1] if self.item_types[-1] != 'new' else f"/data/phi_theta_sample/{self.scene}/train", f"{frame['file_path'].split('/')[-1]}.png")
            self.image_paths += [image_path]
            img = Image.open(image_path)
            img = img.resize(self.img_wh, Image.LANCZOS)
            img = self.transform(img)  # (4, h, w)
            img = img.view(4, -1).permute(1, 0)  # (h*w, 4) RGBA
            self.all_masks += [img[:,-1:]>0]
            img = img[:, :3] * img[:, -1:] + (1 - img[:, -1:])  # blend A to RGB
            self.all_rgbs += [img]

            rays_o, rays_d = get_rays(self.directions, c2w)  # both (h*w, 3)

            self.all_rays += [torch.cat([rays_o, rays_d,
                                         self.near * torch.ones_like(rays_o[:, :1]),
                                         self.far * torch.ones_like(rays_o[:, :1])],
                                        1)]  # (h*w, 8)
            self.all_masks += []

        self.poses = np.stack(self.poses)
        if 'train' == self.split:
            self.all_rays = torch.cat(self.all_rays, 0)  # (len(self.meta['frames])*h*w, 3)
            self.all_rgbs = torch.cat(self.all_rgbs, 0)  # (len(self.meta['frames])*h*w, 3)
        else:
            self.all_rays = torch.stack(self.all_rays, 0)  # (len(self.meta['frames]),h*w, 3)
            self.all_rgbs = torch.stack(self.all_rgbs, 0).reshape(-1,*self.img_wh[::-1], 3)  # (len(self.meta['frames]),h,w,3)
            self.all_masks = torch.stack(self.all_masks, 0).reshape(-1,*self.img_wh[::-1])  # (len(self.meta['frames]),h,w,3)

    def read_source_views(self, file=f"transforms_train.json", pair_idx=None, device=torch.device("cpu")):
        with open(os.path.join(self.root_dir, file), 'r') as f:
            meta = json.load(f)

        w, h = self.img_wh
        focal = 0.5 * 800 / np.tan(0.5 * meta['camera_angle_x'])  # original focal length
        focal *= self.img_wh[0] / 800  # modify focal length to match size self.img_wh

        src_transform = T.Compose([
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])

        # if do not specify source views, load index from pairing file
        if pair_idx is None:
            name = os.path.basename(self.root_dir)
            pair_idx = torch.load('configs/pairs.th')[f'{name}_train'][:3]
            print(f'====> ref idx: {pair_idx}')

        imgs, proj_mats = [], []
        intrinsics, c2ws, w2cs = [],[],[]
        for i, idx in enumerate(pair_idx):
            frame = meta['frames'][idx]
            c2w = np.array(frame['transform_matrix']) @ self.blender2opencv
            w2c = np.linalg.inv(c2w)
            c2ws.append(c2w)
            w2cs.append(w2c)

            # build proj mat from source views to ref view
            proj_mat_l = np.eye(4)
            intrinsic = np.array([[focal, 0, w / 2], [0, focal, h / 2], [0, 0, 1]])
            intrinsics.append(intrinsic.copy())
            intrinsic[:2] = intrinsic[:2] / 4  # 4 times downscale in the feature space
            proj_mat_l[:3, :4] = intrinsic @ w2c[:3, :4]
            if i == 0:  # reference view
                ref_proj_inv = np.linalg.inv(proj_mat_l)
                proj_mats += [np.eye(4)]
            else:
                proj_mats += [proj_mat_l @ ref_proj_inv]

            image_path = os.path.join(self.root_dir, f"{frame['file_path']}.png")
            img = Image.open(image_path)
            img = img.resize(self.img_wh, Image.LANCZOS)
            img = self.transform(img)  # (4, h, w)
            img = img[:3] * img[-1:] + (1 - img[-1:])  # blend A to RGB
            imgs.append(src_transform(img))

        pose_source = {}
        pose_source['c2ws'] = torch.from_numpy(np.stack(c2ws)).float().to(device)
        pose_source['w2cs'] = torch.from_numpy(np.stack(w2cs)).float().to(device)
        pose_source['intrinsics'] = torch.from_numpy(np.stack(intrinsics)).float().to(device)

        near_far_source = [2.0,6.0]
        imgs = torch.stack(imgs).float().unsqueeze(0).to(device)
        proj_mats = torch.from_numpy(np.stack(proj_mats)[:,:3]).float().unsqueeze(0).to(device)
        return imgs, proj_mats, near_far_source, pose_source

    def load_poses_all(self, file=f"transforms_train.json"):
        with open(os.path.join(self.root_dir, file), 'r') as f:
            meta = json.load(f)

        c2ws = []
        for i,frame in enumerate(meta['frames']):
            c2ws.append(np.array(frame['transform_matrix']) @ self.blender2opencv)
        return np.stack(c2ws)

    def define_transforms(self):
        self.transform = T.ToTensor()

    def __len__(self):
        if self.split == 'train':
            return len(self.all_rays)
        return len(self.all_rgbs)

    def __getitem__(self, idx):

        if self.split == 'train':  # use data in the buffers
            # view, ray_idx = torch.randint(0,len(self.all_rays),(1,)), torch.randperm(self.all_rays.shape[1])[:self.args.batch_size]
            # sample = {'rays': self.all_rays[view,ray_idx],
            #           'rgbs': self.all_rgbs[view,ray_idx]}
            sample = {'rays': self.all_rays[idx],
                      'rgbs': self.all_rgbs[idx]}

        else:  # create data for each image separately
            # frame = self.meta['frames'][idx]
            # c2w = torch.FloatTensor(frame['transform_matrix']) @ self.blender2opencv

            img = self.all_rgbs[idx]
            rays = self.all_rays[idx]
            mask = self.all_masks[idx]      # for quantity evaluation
            is_val_item = self.item_types[idx]
            sample = {'rays': rays,
                      'rgbs': img,
                      'mask': mask,
                      'img_index': idx,
                      'item_type': is_val_item}
        return sample