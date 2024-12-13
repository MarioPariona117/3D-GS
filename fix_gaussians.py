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
from scene.gaussian_model import GaussianModel

if __name__ == '__main__':
    model_path = ""
    gaussians = GaussianModel(1)
    path = os.path.join(model_path, "point_cloud", "iteration_30000", "point_cloud.ply")
    output_path = os.path.join(model_path, "point_cloud", "iteration_30000", "point_cloud.ply")
    gaussians.reduce_old_ply(path, output_path, True)
    print(f"{os.path.getsize(path)}")
    print(f"{os.path.getsize(output_path)}")