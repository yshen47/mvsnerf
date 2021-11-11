import numpy as np
import json
import os
import scipy
import copy
from scipy.stats import norm


def bo_for_next_view(scene, prev_indices, n_runs=10, gamma=0.2, lam_kde=20, lam_gp=50, scatter_offset=1.02):
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
        [int(folder.split('_')[-1]) for folder in os.listdir(f'/mvsnerf/runs_fine_tuning/{scene}-ft') if
         'version' in folder], reverse=True)[0]
    mvs_nerf_experiment_folder = os.path.join(f'/mvsnerf/runs_fine_tuning/{scene}-ft/version_{last_version}')

    def load_psnrs(path):
        if os.path.exists(path):
            with open(path, 'r') as f:
                psnrs = json.load(f)
                for k in psnrs:
                    psnrs[k] = np.round(float(psnrs[k]), 4)
        else:
            psnrs = {}
        return psnrs

    orig_train_set_psnr_pth = os.path.join(mvs_nerf_experiment_folder, 'psnrs_train.json')
    orig_train_set_psnrs = load_psnrs(orig_train_set_psnr_pth)

    orig_val_set_psnr_pth = os.path.join(mvs_nerf_experiment_folder, 'psnrs_val.json')
    orig_val_set_psnrs = load_psnrs(orig_val_set_psnr_pth)

    new_set_psnr_pth = os.path.join(mvs_nerf_experiment_folder, 'psnrs_new.json')
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

    k_to_removed = []
    for k in orig_val_set_transforms:
        if k not in orig_val_set_psnrs:
            k_to_removed.append(k)

    for k in k_to_removed:
        del orig_val_set_transforms[k]

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

    train_key_order = list(merged_train_set_transforms.keys())
    train_key_order.sort()
    xyz = prepare_cam_centers(merged_train_set_transforms, train_key_order)

    perf = normalize_psnrs(merged_train_psnrs, train_key_order)

    new_set_key_order = list(new_set_transforms.keys())
    new_set_key_order.sort()
    xyz_next = prepare_cam_centers(new_set_transforms, new_set_key_order)

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
        solved = scipy.linalg.solve(C11, C12, assume_a='pos').T
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
        ind = np.argmin(E_final)
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

