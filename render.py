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

from concurrent.futures import ThreadPoolExecutor, as_completed
import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False


# Parallelisable code for rendering
def render_and_save(idx, view, gaussians, pipeline, background, render_path, gts_path, train_test_exp, separate_sh):
    rendering = render(view, gaussians, pipeline, background, use_trained_exp=train_test_exp, separate_sh=separate_sh)["render"]
    gt = view.original_image[0:3, :, :]

    if train_test_exp:
        rendering = rendering[..., rendering.shape[-1] // 2:]
        gt = gt[..., gt.shape[-1] // 2:]

    render_file = os.path.join(render_path, '{0:05d}.png'.format(idx))
    gt_file = os.path.join(gts_path, '{0:05d}.png'.format(idx))
    torchvision.utils.save_image(rendering, render_file)
    torchvision.utils.save_image(gt, gt_file)

    return idx


def render_set(model_path, name, iteration, views, gaussians, pipeline, background, train_test_exp, separate_sh):
    if len(views) == 0:
        return
    
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    threads = min(len(views), os.cpu_count())
    print(f"Using {threads} threads for rendering")
    with ThreadPoolExecutor(max_workers=threads) as executor:
        futures = {
            executor.submit(
                render_and_save, idx, view, gaussians, pipeline, background, render_path, gts_path, train_test_exp, separate_sh
            ): idx
            for idx, view in enumerate(views)
        }
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Rendering progress"):
            try:
                _ = future.result()
            except Exception as e:
                print(f"Error occurred for index {futures[future]}: {e}")


def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, separate_sh: bool):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
             render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, dataset.train_test_exp, separate_sh)

        if not skip_test:
             render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, dataset.train_test_exp, separate_sh)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, SPARSE_ADAM_AVAILABLE)