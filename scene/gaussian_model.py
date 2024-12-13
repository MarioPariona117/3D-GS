#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import numpy as np
from arguments import OptimizationParams
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn, optim, select
import os
import json
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation


try:
    from diff_gaussian_rasterization import SparseGaussianAdam
except:
    pass

class GaussianModel:
    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize

    def get_actual_covariances (self, selected_pts_mask, scaling_modifier = 1):
        L = build_scaling_rotation(scaling_modifier * self.get_scaling[selected_pts_mask], self._rotation[selected_pts_mask])
        return L @ L.transpose(1, 2)

    def __init__(self, sh_degree, optimizer_type="default", growth_directions_count = 128):
        self.active_sh_degree = 0
        self.optimizer_type = optimizer_type
        self.max_sh_degree = sh_degree  
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions()

        self._growth_directions_count = growth_directions_count
        self._growth_directions = torch.empty(0)
        self._growth_directions_probabilities = torch.empty(0)
        self._growth_length_s = torch.empty(0)
        self._s_prime = torch.empty(0)
        self._v = torch.empty(0)
        self.d_xyz_d_s_prime = torch.empty(0)
        self.d_xyz_d_v = torch.empty(0)
        self.newly_split = torch.empty(0)
        self.just_cloned_mask = torch.empty(0)
        self.newly_cloned = torch.empty(0)

    def create_from_pcd(self, pcd : BasicPointCloud, cam_infos : int, spatial_lr_scale : float):
        """
        Initialises all learnable parameters and other required state based on a point cloud.
        """
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        initialisation_points_count = fused_point_cloud.shape[0]

        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((initialisation_points_count, 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", initialisation_points_count)

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)

        rots = torch.zeros((initialisation_points_count, 4), device="cuda")
        rots[:, 0] = 1
        opacities = self.inverse_opacity_activation(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        # The positions of the "mean" centres of each Gaussian
        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))

        # Learnable colour features
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))

        # Other features
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))

        # Size, rotation, opacity, exposure of each Gaussian
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((initialisation_points_count), device="cuda")
        self.exposure_mapping = {cam_info.image_name: idx for idx, cam_info in enumerate(cam_infos)}
        self.pretrained_exposures = None
        exposure = torch.eye(3, 4, device="cuda")[None].repeat(len(cam_infos), 1, 1)
        self._exposure = nn.Parameter(exposure.requires_grad_(True))
        
        # TODO: Explain masks

        # Learnable parameters for cloning operations
        self.initialise_epo_clone(initialisation_points_count)
        self.initialise_epo_split(initialisation_points_count)

    def initialise_epo_clone(self, initialisation_points_count):
        # Uniform set of discrete growth directions
        phi = torch.linspace(0, 2 * np.pi, self._growth_directions_count, device="cuda")  # Azimuthal angle
        theta = torch.acos(2 * torch.linspace(0, 1, self._growth_directions_count, device="cuda") - 1)  # Polar angle
        
        x = torch.sin(theta) * torch.cos(phi)
        y = torch.sin(theta) * torch.sin(phi)
        z = torch.cos(theta)
        
        self._growth_directions = torch.stack((x, y, z), dim=1)
        self._growth_directions = torch.nn.functional.normalize(self._growth_directions, p=2, dim=1)

        # Initialise growth probabilities uniformly
        self._growth_directions_probabilities = nn.Parameter(torch.full(
            [initialisation_points_count, self._growth_directions_count], 1 / self._growth_directions_count, device="cuda", requires_grad=True)
        )

        # TODO: Why do we start with 1/100
        self._growth_length_s = nn.Parameter(torch.full([initialisation_points_count, 1], 0.0, device="cuda", requires_grad=True))
        
        self.just_cloned_mask = torch.zeros(initialisation_points_count, device = "cuda", dtype = torch.bool)
        self.newly_cloned = torch.zeros(initialisation_points_count, device = "cuda", dtype = torch.bool)

        self.d_index_prob_prob = torch.zeros((initialisation_points_count, 128), device = 'cuda', dtype = torch.float)
        self.d_togrow_d_growth_directions_probabilities = torch.zeros((initialisation_points_count, 128, 3), device = 'cuda', dtype = torch.float)
        self.d_togrow_d_growth_length_s = torch.zeros((initialisation_points_count, 1), device = 'cuda', dtype = torch.float)

    def initialise_epo_split(self, initialisation_points_count):
        samples = self._xyz.detach().clone()[torch.randperm(self._xyz.shape[0])[:12000]]
        differences = samples[:,None,:] - samples[None,:,:]
        self.diameter = torch.sqrt(torch.max(torch.sum(differences ** 2, dim = -1)))
        #print(f"{self.diameter}")
        # Learnable parameters for split meanshift (s_prime) and scalar parameter for the scaling factor (v)
        self._s_prime = nn.Parameter(torch.full([initialisation_points_count, 1], 0.0, device="cuda", requires_grad=True))
        self._v = nn.Parameter(torch.full([initialisation_points_count, 1], 0.0, device="cuda", requires_grad=True))
        # Gradients for thos values
        self.d_xyz_d_s_prime = torch.zeros((initialisation_points_count, 1), device = "cuda")
        self.d_xyz_d_v = torch.zeros((initialisation_points_count, 1), device = "cuda")

        self.newly_split = torch.zeros(initialisation_points_count, device = "cuda", dtype = torch.bool)

    def training_setup(self, training_args: OptimizationParams):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_num_points, 1), device="cuda")
        self.denom = torch.zeros((self.get_num_points, 1), device="cuda")

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},

            {'params': [self._s_prime], 'lr': training_args.s_prime_lr, "name": "s_prime"},
            {'params': [self._v], 'lr': training_args.v_lr, "name": "v"},
            {'params': [self._growth_directions_probabilities], 'lr': training_args.growth_lr, "name": "growth_directions_probabilities"},
            {'params': [self._growth_length_s], 'lr': training_args.growth_length_lr, "name": "growth_length_s"}
        ]

        if self.optimizer_type == "default":
            self.optimizer = optim.Adam(l, lr=0.0, eps=1e-15)
        elif self.optimizer_type == "sparse_adam":
            try:
                self.optimizer = SparseGaussianAdam(l, lr=0.0, eps=1e-15)
            except:
                # A special version of the rasterizer is required to enable sparse adam
                self.optimizer = optim.Adam(l, lr=0.0, eps=1e-15)

        self.exposure_optimizer = optim.Adam([self._exposure])

        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)
        
        self.exposure_scheduler_args = get_expon_lr_func(training_args.exposure_lr_init, training_args.exposure_lr_final,
                                                        lr_delay_steps=training_args.exposure_lr_delay_steps,
                                                        lr_delay_mult=training_args.exposure_lr_delay_mult,
                                                        max_steps=training_args.iterations)

        def custom_lr_func(initial_lr, final_lr, lr_delay_steps, lr_delay_mult, max_steps):
            def lr_func(step):
                if step < lr_delay_steps:
                    # Linear interpolation during the delay phase
                    delay_factor = lr_delay_mult + (1 - lr_delay_mult) * (step / lr_delay_steps)
                    return initial_lr * delay_factor
                else:
                    # Start exponential decay from the end of the delay phase
                    effective_initial_lr = initial_lr * (lr_delay_mult + (1 - lr_delay_mult))
                    decay_rate = (final_lr / effective_initial_lr) ** (1 / (max_steps - lr_delay_steps))
                    adjusted_step = step - lr_delay_steps
                    return effective_initial_lr * (decay_rate ** adjusted_step)

            return lr_func

        self.growth_length_s_scheduler_args = custom_lr_func(
            training_args.growth_length_lr, training_args.growth_length_lr/100,
            lr_delay_steps=3000,
            lr_delay_mult=0.5,
            max_steps=training_args.densify_until_iter
        )

        self.v_scheduler_args = custom_lr_func(
            training_args.v_lr, training_args.v_lr/100,
            lr_delay_steps=3000,
            lr_delay_mult=0.5,
            max_steps=training_args.densify_until_iter
        )

        self.s_prime_scheduler_args = custom_lr_func(
            training_args.s_prime_lr, training_args.s_prime_lr/100,
            lr_delay_steps=3000,
            lr_delay_mult=0.5,
            max_steps=training_args.densify_until_iter
        )

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        if self.pretrained_exposures is None:
            for param_group in self.exposure_optimizer.param_groups:
                param_group['lr'] = self.exposure_scheduler_args(iteration)

        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr

            if param_group["name"] == "growth_length_s":
                lr = self.growth_length_s_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "v":
                lr = self.v_scheduler_args(iteration)
                param_group['lr'] = lr
                # print(f"\n v learning rate is now {lr}")
            if param_group["name"] == "s_prime":
                lr = self.s_prime_scheduler_args(iteration)
                param_group['lr'] = lr


    # TODO: Add EPO variables to capture and restore
    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )
    
    def restore(self, model_args, training_args):
        (self.active_sh_degree, 
        self._xyz, 
        self._features_dc, 
        self._features_rest,
        self._scaling, 
        self._rotation, 
        self._opacity,
        self.max_radii2D, 
        xyz_gradient_accum, 
        denom,
        opt_dict, 
        self.spatial_lr_scale) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    @property
    def get_num_points(self):
        return self._xyz.shape[0]

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_features_dc(self):
        return self._features_dc
    
    @property
    def get_features_rest(self):
        return self._features_rest
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    @property
    def get_exposure(self):
        return self._exposure

    def get_exposure_from_name(self, image_name):
        if self.pretrained_exposures is None:
            return self.get_exposure[self.exposure_mapping[image_name]]
        else:
            return self.pretrained_exposures[image_name]
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def reset_opacity(self):
        opacities_new = self.inverse_opacity_activation(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append(f'f_dc_{i}')
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append(f'f_rest_{i}')
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append(f'scale_{i}')
        for i in range(self._rotation.shape[1]):
            l.append(f'rot_{i}')
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(self.get_num_points, dtype=dtype_full)

        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def load_ply(self, path, use_train_test_exp = False):
        plydata = PlyData.read(path)
        if use_train_test_exp:
            exposure_file = os.path.join(os.path.dirname(path), os.pardir, os.pardir, "exposure.json")
            if os.path.exists(exposure_file):
                with open(exposure_file, "r") as f:
                    exposures = json.load(f)
                self.pretrained_exposures = {image_name: torch.FloatTensor(exposures[image_name]).requires_grad_(False).cuda() for image_name in exposures}
                print(f"Pretrained exposures loaded.")
            else:
                print(f"No exposure to be loaded at {exposure_file}")
                self.pretrained_exposures = None

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors
    
    def cat_tensors_to_optimizer(self, tensors_dict):
        """
        Adds corresponding tensors to all the groups in self.optimizer.param_groups.
        """
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)
                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state
                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]
                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state
                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self._s_prime = optimizable_tensors["s_prime"]
        self._v = optimizable_tensors["v"]
        self._growth_directions_probabilities = optimizable_tensors['growth_directions_probabilities']
        self._growth_length_s = optimizable_tensors['growth_length_s']

        self.newly_split = self.newly_split[valid_points_mask]
        self.newly_cloned = self.newly_cloned[valid_points_mask]
        self.just_cloned_mask = self.just_cloned_mask[valid_points_mask]

        self.d_xyz_d_s_prime = self.d_xyz_d_s_prime[valid_points_mask]
        self.d_xyz_d_v = self.d_xyz_d_v[valid_points_mask]
        self.d_index_prob_prob = self.d_index_prob_prob[valid_points_mask]
        self.d_togrow_d_growth_directions_probabilities = self.d_togrow_d_growth_directions_probabilities[valid_points_mask]
        self.d_togrow_d_growth_length_s = self.d_togrow_d_growth_length_s[valid_points_mask]


        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]
        self.tmp_radii = self.tmp_radii[valid_points_mask]
    
    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1
    
    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size, radii, iteration, epo_start_iteration=4500):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.tmp_radii = radii
        if iteration >= epo_start_iteration:
            self.densify_and_clone(grads, max_grad, extent)
            self.densify_and_split(grads, max_grad, extent)
        else:
            self.densify_and_clone_heuristic(grads, max_grad, extent)
            self.densify_and_split_heuristic(grads, max_grad, extent)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)
        self.tmp_radii = None

        torch.cuda.empty_cache()

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_tmp_radii, new_s_prime, new_v, new_growth_directions_probabilities, new_growth_length_s, new_newly_split, new_newly_cloned):
        """Maintain tensor size equality
        
        This function takes in a list of newly added points (their corresponding tensors) and concats them onto the original tensors.
        """
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation,
        "s_prime": new_s_prime,
        "v": new_v,
        'growth_directions_probabilities' : new_growth_directions_probabilities,
        'growth_length_s' : new_growth_length_s}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self._s_prime = optimizable_tensors["s_prime"]
        self._v = optimizable_tensors["v"]
        self._growth_directions_probabilities = optimizable_tensors['growth_directions_probabilities']
        self._growth_length_s = optimizable_tensors['growth_length_s']

        self.newly_split = torch.cat((self.newly_split, new_newly_split), dim = 0)
        self.just_cloned_mask = torch.cat((self.just_cloned_mask, torch.zeros(new_xyz.size()[0], device = "cuda", dtype = torch.bool)))
        self.newly_cloned = torch.cat((self.newly_cloned, new_newly_cloned), dim = 0)

        self.d_index_prob_prob = torch.cat((self.d_index_prob_prob, torch.zeros((new_xyz.shape[0], 128), device="cuda", dtype=torch.float)))
        self.d_togrow_d_growth_directions_probabilities = torch.cat((self.d_togrow_d_growth_directions_probabilities, torch.zeros((new_xyz.shape[0], 128, 3), device="cuda", dtype=torch.float)))
        self.d_togrow_d_growth_length_s = torch.cat((self.d_togrow_d_growth_length_s,torch.zeros((new_xyz.shape[0], 1), device="cuda", dtype=torch.float)))

        self.tmp_radii = torch.cat((self.tmp_radii, new_tmp_radii))
        self.xyz_gradient_accum = torch.zeros((self.get_num_points, 1), device="cuda")
        self.denom = torch.zeros((self.get_num_points, 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_num_points), device="cuda")
    
    def del_mu(self, selected_pts_mask):
        # Get matrix from quaternion.
        rots = build_rotation(self._rotation[selected_pts_mask])

        stds = self.get_scaling[selected_pts_mask]

        S = stds / (1 + torch.exp(-self._s_prime[selected_pts_mask]))

        # First half is for del_mu, second half is for -del_mu
        # Multiply by 1+eps to preserve gradients (zeroes out otherwise)
        
        return torch.bmm(rots, S.unsqueeze(-1)).squeeze(-1)

    def densify_and_split(self, grads, grad_threshold, scene_extent, eps=1e-3):
        """Perform EPO split
        
        Chooses points that satisfy the gradient condition.
        For each, creates 2 points (as opposed to N in 3DGS).
        Uses 𝜇 +- 𝛿𝜇𝑘 for the new position. 𝛿𝜇𝑘 =𝑅(𝜎𝑘∗(1/1+𝑒𝑥𝑝(−𝑠′)))
        Uses exp(log(scale) / 𝜙) for the new Gaussian scaling. 𝜙 = 1.2∗(1/1+𝑒𝑥𝑝(−𝑣))+1
        """
        n_init_points = self.get_num_points
        
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)

        # Create new points and find d(new_xyz)/d(s_prime)
        x = self.del_mu(selected_pts_mask)
        x = torch.cat((x*(1+eps), -x))
        new_xyz = x + self.get_xyz[selected_pts_mask].repeat(2, 1)
        new_xyz.backward(torch.ones_like(new_xyz))
        self.d_xyz_d_s_prime = torch.concat((self._s_prime.grad, self._s_prime.grad[selected_pts_mask].repeat(2, 1)))
        new_s_prime = self._s_prime[selected_pts_mask].repeat(2, 1)

        new_v = self._v[selected_pts_mask].repeat(2, 1)
        phi = 1.2 * (1/(1 + torch.exp(-new_v))) + 1
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(2,1) / phi)

        new_scaling.backward(torch.ones_like(new_scaling))
        self.d_xyz_d_v = torch.concat((self._v.grad, self._v.grad[selected_pts_mask].repeat(2, 1)))

        # maintain rotation, features, opacity, radii, s_prime
        new_rotation = self._rotation[selected_pts_mask].repeat(2,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(2,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(2,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(2,1)
        new_tmp_radii = self.tmp_radii[selected_pts_mask].repeat(2)
        new_growth_directions_probabilities = self._growth_directions_probabilities[selected_pts_mask].repeat(2 ,1)
        new_growth_length_s = self._growth_length_s[selected_pts_mask].repeat(2, 1)

        added_points_count = new_xyz.size()[0]

        new_newly_cloned = torch.zeros(added_points_count, device = "cuda", dtype = torch.bool)
        new_newly_split = torch.ones(added_points_count, device = "cuda", dtype = torch.bool)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation, new_tmp_radii, new_s_prime, new_v, new_growth_directions_probabilities, new_growth_length_s, new_newly_split, new_newly_cloned)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(added_points_count, device="cuda", dtype=torch.bool)))
        self.prune_points(prune_filter)
        
    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        """Perform EPO clone
        
        TODO: Insert formulae here.
        """
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        #print(f"{self._growth_length_s.mean()}")
        growth_dist = self.calc_growth_dist(selected_pts_mask)
        softmax_growth_dir = self.calc_growth_dir_soft(selected_pts_mask)
        argmax_growth_dir = self.calc_growth_dir_repara(selected_pts_mask)

        # Reparametrisation trick - by detaching some tensors, they are considered as constants.
        # Gradients for reparameterised_dir are equal to that of differentiable softmax_growth_dir
        # But the value of reparameterised_dir is equal to the non-differentiable argmax_growth_dir
        reparameterised_dir = argmax_growth_dir.detach() + softmax_growth_dir - softmax_growth_dir.detach()
        togrow = torch.mul(growth_dist, reparameterised_dir)

        new_xyz = self._xyz[selected_pts_mask] + togrow
        added_pts = new_xyz.size()[0]

        togrow.backward(torch.ones_like(togrow), retain_graph=True)
        # N x 128 x 3
        self.d_togrow_d_growth_directions_probabilities = self.index_directions.grad.detach().clone()
        self.d_togrow_d_growth_length_s = self._growth_length_s.grad.detach().clone()
        self._growth_directions_probabilities.grad = None
        self.index_directions.grad = None
        self._growth_length_s.grad = None
        self.index_directions.backward(torch.ones_like(self.index_directions))
        self.d_index_prob_prob = self._growth_directions_probabilities.grad

        new_newly_cloned = torch.ones(added_pts, device = "cuda", dtype = torch.bool)
        self.just_cloned_mask = selected_pts_mask
        
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]
        new_tmp_radii = self.tmp_radii[selected_pts_mask]

        new_s_prime = self._s_prime[selected_pts_mask]
        new_v = self._v[selected_pts_mask]
        new_growth_directions_probabilities = self._growth_directions_probabilities[selected_pts_mask]
        new_growth_length_s = self._growth_length_s[selected_pts_mask]

        new_newly_split = torch.zeros(added_pts, device = "cuda", dtype = torch.bool)
        self.d_xyz_d_s_prime = torch.concat((self.d_xyz_d_s_prime, torch.zeros((added_pts, 1), device = "cuda", dtype=torch.bool)))
        self.d_xyz_d_v = torch.concat((self.d_xyz_d_v, torch.zeros((added_pts, 1), device = "cuda", dtype=torch.bool)))

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_tmp_radii, new_s_prime, new_v, new_growth_directions_probabilities, new_growth_length_s, new_newly_split, new_newly_cloned)

    def clone_handle_gradients (self, togrow, newsize, selected_pts_mask):
        # TODO: As stated above, change this to be cleaner
        togrow.backward(torch.ones_like(togrow), retain_graph=True)
        # N x 128 x 3
        self.d_togrow_d_growth_directions_probabilities = self.index_directions.grad.detach().clone()
        self.d_togrow_d_growth_length_s = self._growth_length_s.grad.detach().clone()
        self._growth_directions_probabilities.grad = None
        self.index_directions.grad = None
        self._growth_length_s.grad = None
        self.index_directions.backward(torch.ones_like(self.index_directions))
        self.d_index_prob_prob = self._growth_directions_probabilities.grad.detach().clone()

        new_newly_cloned = torch.ones(newsize, device = "cuda", dtype = torch.bool)

        self.just_cloned_mask = selected_pts_mask
        
        return new_newly_cloned

    def calc_growth_dir_soft(self, selected_pts_mask, temperature=1e-3):
        """A differentiable replacement for argmax"""
        index_soft = torch.nn.functional.softmax(self._growth_directions_probabilities / temperature, dim = 1)
        # TODO: Rename and explain index_directions
        self.index_directions = index_soft.unsqueeze(-1) * self._growth_directions
        self.index_directions.retain_grad()
        return self.index_directions[selected_pts_mask].sum(dim=1)

    def calc_growth_dir_repara(self, selected_pts_mask):
        index = torch.argmax(self._growth_directions_probabilities[selected_pts_mask], dim=1)
        index_hard = torch.nn.functional.one_hot(index, num_classes=self._growth_directions.shape[0]).to(self._growth_directions.device)
        return torch.matmul(index_hard.float(), self._growth_directions)
    
    def calc_growth_dist (self, selected_pts_mask):
        # v is 2 * maximum standard deviation of original gaussians
        # max variance = max eigenvalue of covariance matrix
        covariances = self.get_actual_covariances(selected_pts_mask)
        eigvals = torch.linalg.eigvals(covariances)
        eigvals = eigvals.type(torch.float)
        variance = torch.max(eigvals, dim = 1).values
        sd = torch.sqrt(variance).unsqueeze(1)
        ret = 2 * sd / (1 + torch.exp(- self._growth_length_s[selected_pts_mask]))
        # ret = 5e-4 * sd / (1 + torch.exp(- self._growth_length_s[selected_pts_mask]))
        return ret
    
    def calc_evolutive_density_control_param_grads (self):
        self.calc_clone_grads()
        self.calc_split_grads()
        self.newly_cloned = torch.zeros(self.newly_cloned.size(), device = "cuda", dtype = torch.bool)
        self.newly_split = torch.zeros(self.newly_split.size(), device = "cuda", dtype = torch.bool)

    def calc_split_grads(self):
        fresh_xyzprime_grads = self._xyz.grad[self.newly_split].unsqueeze(1)
        fresh_scale_grads = self._scaling.grad[self.newly_split].unsqueeze(1)

        # n x 1 = n x 3 * n x 3 x 1 * n x 1
        d_loss_d_s_prime = torch.zeros((self._xyz.size()[0], 1), device = "cuda")

        # dloss/ds = dloss/dx' * (dx'/ds_x' * ds_x'/ds = dx'/ds)
        # n x 1 x 1 = n x 1 x 3 * n x 3 x 1
        d_loss_d_s_prime[self.newly_split] = torch.matmul(fresh_xyzprime_grads, self.d_xyz_d_s_prime[self.newly_split].expand(-1, 3).unsqueeze(-1)).squeeze(-1)

        self._s_prime.grad = d_loss_d_s_prime

        # n x 1 = n x 3 * n x 3 x 1 * n x 1
        d_loss_d_v = torch.zeros((self._xyz.size()[0], 1), device = "cuda")

        # dloss/ds = dloss/dx' * (dx'/ds_x' * ds_x'/ds = dx'/ds)
        # n x 1 x 1 = n x 1 x 3 * n x 3 x 1
        d_loss_d_v[self.newly_split] = torch.matmul(fresh_scale_grads, self.d_xyz_d_v[self.newly_split].expand(-1, 3).unsqueeze(-1)).squeeze(-1)

        self._v.grad = d_loss_d_v

    def calc_clone_grads (self):
        fresh_xyzprime_grads = self._xyz.grad[self.newly_cloned].unsqueeze(1)

        # n x 128 = n x 3 * n x 3 x 128 * n x 128
        d_loss_d_growth_directions_probabilities = torch.zeros((self._xyz.size()[0], self._growth_directions_count), device = "cuda")

        # dloss/dprob = dloss/dx' * dx'/dprob_x' * dprob_x'/dprob
        # n x 1 x 128 = n x 1 x 3 * n x 3 x 128
        tmp = torch.matmul(fresh_xyzprime_grads, self.d_togrow_d_growth_directions_probabilities[self.just_cloned_mask].permute(0, 2, 1))

        # n x 128 = n x 128 . n x 128
        tmp = tmp.squeeze(1) * self.d_index_prob_prob[self.just_cloned_mask]

        d_loss_d_growth_directions_probabilities[self.just_cloned_mask] = tmp

        self._growth_directions_probabilities.grad = d_loss_d_growth_directions_probabilities

        # n x 1 = n x 3 * n x 3 x 1 * n x 1
        d_loss_d_growth_length_s = torch.zeros((self._xyz.size()[0], 1), device = "cuda")

        # dloss/ds = dloss/dx' * (dx'/ds_x' * ds_x'/ds = dx'/ds)
        # n x 1 x 1 = n x 1 x 3 * n x 3 x 1
        d_loss_d_growth_length_s[self.just_cloned_mask] = torch.matmul(fresh_xyzprime_grads, self.d_togrow_d_growth_length_s[self.just_cloned_mask].expand(-1, 3).unsqueeze(-1)).squeeze(-1)

        self._growth_length_s.grad = d_loss_d_growth_length_s
    
    def densify_and_split_heuristic(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)
        new_tmp_radii = self.tmp_radii[selected_pts_mask].repeat(N)

        new_s_prime = self._s_prime[selected_pts_mask].repeat(N,1)
        new_v = self._v[selected_pts_mask].repeat(N,1)
        new_growth_directions_probabilities = self._growth_directions_probabilities[selected_pts_mask].repeat(N,1)
        new_growth_length_s = self._growth_length_s[selected_pts_mask].repeat(N,1)

        added_points_count = new_xyz.size()[0]

        new_newly_cloned = torch.zeros(added_points_count, device = "cuda", dtype = torch.bool)
        new_newly_split = torch.ones(added_points_count, device = "cuda", dtype = torch.bool)

        self.d_xyz_d_s_prime = torch.concat((self.d_xyz_d_s_prime, self.d_xyz_d_s_prime[selected_pts_mask].repeat(2, 1)))
        self.d_xyz_d_v = torch.concat((self.d_xyz_d_v, self.d_xyz_d_v[selected_pts_mask].repeat(2, 1)))

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation, new_tmp_radii, new_s_prime, new_v, new_growth_directions_probabilities, new_growth_length_s, new_newly_split, new_newly_cloned)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone_heuristic(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        new_tmp_radii = self.tmp_radii[selected_pts_mask]

        new_s_prime = self._s_prime[selected_pts_mask]
        new_v = self._v[selected_pts_mask]
        new_growth_directions_probabilities = self._growth_directions_probabilities[selected_pts_mask]
        new_growth_length_s = self._growth_length_s[selected_pts_mask]

        added_pts = new_xyz.size()[0]

        new_newly_split = torch.zeros(added_pts, device = "cuda", dtype = torch.bool)
        new_newly_cloned = torch.zeros(added_pts, device = "cuda", dtype = torch.bool)

        self.d_xyz_d_s_prime = torch.concat((self.d_xyz_d_s_prime, torch.zeros((added_pts, 1), device = "cuda", dtype=torch.bool)))
        self.d_xyz_d_v = torch.concat((self.d_xyz_d_v, torch.zeros((added_pts, 1), device = "cuda", dtype=torch.bool)))

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_tmp_radii, new_s_prime, new_v, new_growth_directions_probabilities, new_growth_length_s, new_newly_split, new_newly_cloned)
