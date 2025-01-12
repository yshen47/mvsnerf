import random

import numpy as np
import json
import os
import copy
from pathlib import Path
from scipy.stats import norm


# prepare cam_centers (cam2world)
def prepare_cam_centers(transforms, key_order):
    zero = np.array([0, 0, 0])
    new_pts = []
    for k in key_order:
        pose = transforms[k]
        cam_center_in_world = np.matmul(pose[:3, :3], zero) + pose[:3, 3]
        new_pts.append(cam_center_in_world)
    xyz = np.stack(new_pts)
    xyz = xyz / np.expand_dims(np.linalg.norm(xyz, axis=1), axis=1)
    return xyz


def bo_for_next_view(exp_name, scene, prev_indices, avg_val_camera_center, n_runs=5, gamma=0.2, lam_kde=20, lam_gp=50, scatter_offset=1.02, trust_region_radius=0.6):
    def load_transform(path):
        with open(path, 'r') as f:
            orig_transforms = json.load(f)
            res = dict(('/'.join(m['file_path'].split('/')[-2:]), np.array(m['transform_matrix'])) for m in
                       orig_transforms['frames'])
        return res

    orig_train_set_transforms = load_transform(f'/data/original_nerf_synthetic/{scene}/transforms_train.json')
    orig_val_set_transforms = load_transform(f'/data/original_nerf_synthetic/{scene}/transforms_val.json')
    new_set_transforms = load_transform(f'/data/phi_theta_sample/{scene}/transforms_train.json')
    version_num = -1
    last_version = sorted(
        [int(folder.split('_')[-1]) for folder in os.listdir(f'/mvsnerf/runs_fine_tuning/{exp_name}') if
         'version' in folder], reverse=True)[0]
    mvs_nerf_experiment_folder = os.path.join(f'/mvsnerf/runs_fine_tuning/{exp_name}/version_{last_version}')

    def load_psnrs(path):
        if os.path.exists(path):
            with open(path, 'r') as f:
                psnrs = json.load(f)
                for k in psnrs:
                    psnrs[k] = np.round(float(psnrs[k]), 4)
        else:
            psnrs = {}
        return psnrs

    project_root_path = f'/mvsnerf/runs_fine_tuning/{exp_name}'
    curr_psnrs_ind = 0
    root_path = Path(project_root_path)
    if len(list(root_path.glob('psnrs_train_*.json'))) > 0:
        latest_psnrs_train = sorted(root_path.glob('psnrs_train_*.json'))[-1]
        curr_psnrs_ind = int(str(latest_psnrs_train).split('_')[-1][:-5])

    orig_train_set_psnr_pth = os.path.join(mvs_nerf_experiment_folder, 'psnrs_train_{:03d}.json'.format(curr_psnrs_ind))
    orig_train_set_psnrs = load_psnrs(orig_train_set_psnr_pth)

    new_set_psnr_pth = os.path.join(mvs_nerf_experiment_folder, 'psnrs_new_{:03d}.json'.format(curr_psnrs_ind))
    new_set_psnrs = load_psnrs(new_set_psnr_pth)

    # merge original train set and next view train set

    merged_train_set_transforms = copy.deepcopy(orig_train_set_transforms)
    merged_train_set_transforms.update(new_set_transforms)
    merged_train_psnrs = copy.deepcopy(orig_train_set_psnrs)
    merged_train_psnrs.update(new_set_psnrs)

    k_to_removed = []
    for k in merged_train_set_transforms:
        if k not in merged_train_psnrs:
            k_to_removed.append(k)

    for k in k_to_removed:
        del merged_train_set_transforms[k]

    def normalize_psnrs(psnrs, key_order):
        psnrs_list = []
        for k in key_order:
            psnrs_list.append(psnrs[k])
        psnrs_list = np.array(psnrs_list)
        max_psnr = np.max(psnrs_list)
        min_psnr = np.min(psnrs_list)

        norm_psnrs_list = np.clip((np.array(psnrs_list) - min_psnr) / (max_psnr - min_psnr), 0, 1)
        norm_psnrs_list[norm_psnrs_list == 0] = 0.01

        perf = np.array(norm_psnrs_list)
        return perf

    orig_keys = []
    new_keys = []
    new_key_prefix = list(new_set_transforms.keys())[0].split('/')[0]
    orig_key_prefix = 'train'
    for k in list(merged_train_set_transforms.keys()):
        if 'new_results' in k:
            # new_key_prefix = '/'.join(k.split('/')[:-1])
            new_keys.append(int(k.split('/')[-1][2:]))
        else:
            # orig_key_prefix = '/'.join(k.split('/')[:-1])
            orig_keys.append(int(k.split('/')[-1][2:]))
    orig_keys.sort()
    new_keys.sort()
    train_key_order = [orig_key_prefix + "/r_" + str(item) for item in orig_keys] + [new_key_prefix + "/r_" + str(item) for item in new_keys]

    xyz = prepare_cam_centers(merged_train_set_transforms, train_key_order)

    perf = normalize_psnrs(merged_train_psnrs, train_key_order)

    new_set_key_order = [new_key_prefix + "/r_" + str(item) for item in list(np.arange(len(new_set_transforms)))]
    xyz_next = prepare_cam_centers(new_set_transforms, new_set_key_order)
    distances = np.linalg.norm(xyz_next - avg_val_camera_center, axis=1)
    trust_region_valid_indices = np.where(distances < trust_region_radius)[0]
    # spherical gaussian kernel
    def kernel_func(A, B, lam=50):
        # A=Nx3, B=Mx3
        return np.exp(-2 * lam * (1 - A.dot(B.T)))

    # gaussian process
    def GP(X1, y1, X2, lam=50):
        # loss_optimum = np.min(y1)
        # Kernel of the observations
        C11 = kernel_func(X1, X1, lam)
        # Kernel of observations vs to-predict
        C12 = kernel_func(X1, X2, lam)
        # Solve
        solved = np.linalg.lstsq(C11, C12, rcond=None)[0].T # with linalg.solve sometimes Matrix X1 is singular, thus change to np.linalg.lstsq
        # Compute posterior mean
        m2 = solved @ y1
        # Compute the posterior covariance
        C22 = kernel_func(X2, X2)
        C2 = C22 - (solved @ C12)
        return m2, C2

    inds = []
    for prev_index in prev_indices:
        inds.append(int(prev_index.split('_')[-1]))

    while n_runs:
        # run BO
        # print(f'Number of Optimization run: {i}')
        # get initial data ready
        X1 = xyz
        X2 = xyz_next
        y1 = perf - 0.5
        # Caculate trackability
        K = kernel_func(xyz, xyz_next, lam_kde)  # np.exp(-2*lam*(1-xyz.dot(xyz_test.T)))
        E_track = -np.max(K, axis=0)
        # GP
        m2, C2 = GP(X1, y1, X2, lam_gp)
        sigma2 = np.diag(C2)
        # BO
        loss_optimum = np.min(y1)
        Z = - (m2 - loss_optimum) / sigma2
        EI = - (m2 - loss_optimum) * norm.cdf(Z) + sigma2 * norm.pdf(Z)
        EI[sigma2 == 0.0] == 0.0
        EI = -EI
        E_perf = EI
        E_final = E_perf + gamma * E_track
        # Result
        sorted_ind = np.argsort(E_final)
        ind = None
        for index in sorted_ind:
            if index in trust_region_valid_indices:
                ind = index
                break
        if ind not in inds:
            n_runs -= 1
            inds.append(ind)
        xyz_next_sample = np.expand_dims(xyz_next[ind], axis=0)
        print('m2 ind: ', m2[ind])
        psnr_next_sample = m2[ind] + 0.5
        print(f'Candidate sample is: {xyz_next_sample}')
        print(f'Sample psnr is: {psnr_next_sample}')
        # add new samples to existing observations
        xyz = np.concatenate((xyz, xyz_next_sample))
        perf = np.concatenate((perf, np.array([psnr_next_sample])))

    inds = [new_set_key_order[ind].split('/')[-1] for ind in inds]
    inds.sort()
    with open(os.path.join(mvs_nerf_experiment_folder, 'next_view_indices.txt'), 'a+') as f:
        f.write(",".join(inds) + '\n')


def random_for_next_view(exp_name, scene, prev_indices, avg_val_camera_center, n_runs=5, gamma=0.2, lam_kde=20, lam_gp=50, scatter_offset=1.02, trust_region_radius=0.6):

    def load_transform(path):
        with open(path, 'r') as f:
            orig_transforms = json.load(f)
            res = dict(('/'.join(m['file_path'].split('/')[-2:]), np.array(m['transform_matrix'])) for m in
                       orig_transforms['frames'])
        return res

    orig_train_set_transforms = load_transform(f'/data/original_nerf_synthetic/{scene}/transforms_train.json')
    new_set_transforms = load_transform(f'/data/phi_theta_sample/{scene}/transforms_train.json')

    last_version = sorted(
        [int(folder.split('_')[-1]) for folder in os.listdir(f'/mvsnerf/runs_fine_tuning/{exp_name}') if
         'version' in folder], reverse=True)[0]
    mvs_nerf_experiment_folder = os.path.join(f'/mvsnerf/runs_fine_tuning/{exp_name}/version_{last_version}')

    def load_psnrs(path):
        if os.path.exists(path):
            with open(path, 'r') as f:
                psnrs = json.load(f)
                for k in psnrs:
                    psnrs[k] = np.round(float(psnrs[k]), 4)
        else:
            psnrs = {}
        return psnrs

    project_root_path = f'/mvsnerf/runs_fine_tuning/{exp_name}'
    curr_psnrs_ind = 0
    root_path = Path(project_root_path)
    if len(list(root_path.glob('psnrs_train_*.json'))) > 0:
        latest_psnrs_train = sorted(root_path.glob('psnrs_train_*.json'))[-1]
        curr_psnrs_ind = int(str(latest_psnrs_train).split('_')[-1][:-5])

    orig_train_set_psnr_pth = os.path.join(mvs_nerf_experiment_folder, 'psnrs_train_{:03d}.json'.format(curr_psnrs_ind))
    orig_train_set_psnrs = load_psnrs(orig_train_set_psnr_pth)

    new_set_psnr_pth = os.path.join(mvs_nerf_experiment_folder, 'psnrs_new_{:03d}.json'.format(curr_psnrs_ind))
    new_set_psnrs = load_psnrs(new_set_psnr_pth)

    # merge original train set and next view train set

    merged_train_set_transforms = copy.deepcopy(orig_train_set_transforms)
    merged_train_set_transforms.update(new_set_transforms)
    merged_train_psnrs = copy.deepcopy(orig_train_set_psnrs)
    merged_train_psnrs.update(new_set_psnrs)

    k_to_removed = []
    for k in merged_train_set_transforms:
        if k not in merged_train_psnrs:
            k_to_removed.append(k)

    for k in k_to_removed:
        del merged_train_set_transforms[k]

    new_key_prefix = list(new_set_transforms.keys())[0].split('/')[0]
    new_set_key_order = [new_key_prefix + "/r_" + str(item) for item in list(np.arange(len(new_set_transforms)))]
    # new_set_key_order = ["/r_" + str(item) for item in list(np.arange(len(new_set_transforms)))]
    xyz_next = prepare_cam_centers(new_set_transforms, new_set_key_order)
    inds = []
    distances = np.linalg.norm(xyz_next - avg_val_camera_center, axis=1)
    trust_region_valid_indices = np.where(distances < trust_region_radius)[0]

    candidates = list(trust_region_valid_indices)
    for prev_index in prev_indices:
        inds.append(int(prev_index.split('_')[-1]))
        if inds[-1] in candidates:
            candidates.remove(inds[-1])
    inds += random.choices(candidates, k=n_runs)
    inds = [new_set_key_order[ind].split('/')[-1] for ind in inds]
    inds.sort()
    with open(os.path.join(mvs_nerf_experiment_folder, 'next_view_indices.txt'), 'a+') as f:
        f.write(",".join(inds) + '\n')