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

import os
from argparse import ArgumentParser
import time

mipnerf360_outdoor_scenes = ["bicycle", "stump"]    
mipnerf360_indoor_scenes = ["counter"]
tanks_and_temples_scenes = ["truck", "train"]
deep_blending_scenes = ["playroom"]
llff_scenes = ["horns", "trex"]
mipnerf360_outdoor_scenes = ["bicycle", "flowers", "garden", "stump", "treehill"]
mipnerf360_indoor_scenes = ["room", "counter", "kitchen", "bonsai"]
tanks_and_temples_scenes = ["truck", "train"]
deep_blending_scenes = ["drjohnson", "playroom"]
llff_scenes = ["fern", "flower", "fortress", "horns", "leaves", "orchids", "room", "trex"]

parser = ArgumentParser(description="Full evaluation script parameters")
parser.add_argument("--skip_training", action="store_true")
parser.add_argument("--skip_rendering", action="store_true")
parser.add_argument("--skip_metrics", action="store_true")
parser.add_argument("--output_path", default="./eval")
parser.add_argument("--use_depth", action="store_true")
parser.add_argument("--use_expcomp", action="store_true")
parser.add_argument("--fast", action="store_true")
parser.add_argument("--aa", action="store_true")

args, _ = parser.parse_known_args()

all_scenes = []
all_scenes.extend(mipnerf360_outdoor_scenes)
all_scenes.extend(mipnerf360_indoor_scenes)
all_scenes.extend(tanks_and_temples_scenes)
all_scenes.extend(deep_blending_scenes)
all_scenes.extend(llff_scenes)

if not args.skip_training or not args.skip_rendering:
    parser.add_argument('--mipnerf360', "-m360", required=True, type=str)
    parser.add_argument("--tanksandtemples", "-tat", required=True, type=str)
    parser.add_argument("--deepblending", "-db", required=True, type=str)
    parser.add_argument("--llff", "-lf", required=True, type=str, help="Path to the LLFF dataset")

    args = parser.parse_args()
if not args.skip_training:
    common_args = " --disable_viewer --quiet --eval --test_iterations -1 "
    
    if args.aa:
        common_args += " --antialiasing "
    if args.use_depth:
        common_args += " -d depths2/ "

    if args.use_expcomp:
        common_args += " --exposure_lr_init 0.001 --exposure_lr_final 0.0001 --exposure_lr_delay_steps 5000 --exposure_lr_delay_mult 0.001 --train_test_exp "

    if args.fast:
        common_args += " --optimizer_type sparse_adam "

    start_time = time.time()
    for scene in mipnerf360_outdoor_scenes:
        source = args.mipnerf360 + "/" + scene
        s_time = time.time()
        os.system("python train.py -s " + source + " -i images_8 -m " + args.output_path + "/" + scene + common_args)
        with open(os.path.join(args.output_path,scene,"timing.txt"), 'w') as file:
            file.write(f"{time.time() - s_time}")
            
    for scene in mipnerf360_indoor_scenes:
        source = args.mipnerf360 + "/" + scene
        s_time = time.time()
        os.system("python train.py -s " + source + " -i images_2 -m " + args.output_path + "/" + scene + common_args)
        with open(os.path.join(args.output_path,scene,"timing.txt"), 'w') as file:
            file.write(f"{(time.time() - s_time) / 60.0}")
    m360_timing = (time.time() - start_time)/60.0

    start_time = time.time()
    for scene in tanks_and_temples_scenes:
        source = args.tanksandtemples + "/" + scene
        s_time = time.time()
        os.system("python train.py -s " + source + " -m " + args.output_path + "/" + scene + common_args)
        with open(os.path.join(args.output_path,scene,"timing.txt"), 'w') as file:
            file.write(f"{(time.time() - s_time) / 60.0}")
    tandt_timing = (time.time() - start_time)/60.0

    start_time = time.time()
    for scene in deep_blending_scenes:
        source = args.deepblending + "/" + scene
        s_time = time.time()
        os.system("python train.py -s " + source + " -m " + args.output_path + "/" + scene + common_args)
        with open(os.path.join(args.output_path,scene,"timing.txt"), 'w') as file:
            file.write(f"{(time.time() - s_time) / 60.0}")
    db_timing = (time.time() - start_time)/60.0

    start_time = time.time()
    for scene in llff_scenes:
        # print(scene)
        source = os.path.join(args.llff, scene) 
        s_time = time.time()
        common_args_llff = common_args 
        os.system("python train.py -s " + source + " -i images -m " + args.output_path + "/" + scene + common_args_llff)
        with open(os.path.join(args.output_path,scene,"timing.txt"), 'w') as file:
            file.write(f"{(time.time() - s_time) / 60.0}")
    llff_timing = (time.time() - start_time) / 60.0

    with open(os.path.join(args.output_path,"timing.txt"), 'w') as file:
        file.write(f"m360: {m360_timing} minutes \n tandt: {tandt_timing} minutes \n db: {db_timing} minutes\n")

if not args.skip_rendering:
    all_sources = []
    for scene in mipnerf360_outdoor_scenes:
        all_sources.append(args.mipnerf360 + "/" + scene)
    for scene in mipnerf360_indoor_scenes:
        all_sources.append(args.mipnerf360 + "/" + scene)
    for scene in tanks_and_temples_scenes:
        all_sources.append(args.tanksandtemples + "/" + scene)
    for scene in deep_blending_scenes:
        all_sources.append(args.deepblending + "/" + scene)
    for scene in llff_scenes:
        all_sources.append(os.path.join(args.llff, scene))
    
    common_args = " --quiet --eval --skip_train"
    
    if args.aa:
        common_args += " --antialiasing "
    if args.use_expcomp:
        common_args += " --train_test_exp "

    for scene, source in zip(all_scenes, all_sources):
        os.system("python render.py --iteration 7000 -s " + source + " -m " + args.output_path + "/" + scene + common_args)
        os.system("python render.py --iteration 30000 -s " + source + " -m " + args.output_path + "/" + scene + common_args)

if not args.skip_metrics:
    scenes_string = ""
    for scene in all_scenes:
        scenes_string += "\"" + args.output_path + "/" + scene + "\" "

    os.system("python metrics.py -m " + scenes_string)
