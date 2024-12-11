import json
import os
print("Current working directory:", os.getcwd())

# SCENES = ["stump", "counter", "playroom", "horns", "trex"]
# DIRECTORY = {
#     "MipNeRF360 Outdoor": ["stump"],
#     "MipNeRF360 Indoor": ["counter"],
#     "Tanks & Temples": [],
#     "Deep Blending": ["playroom"],
#     "LLFF": ["horns", "trex"]
# }
DIRECTORY = {
    "MipNeRF360 Outdoor": ["bicycle", "stump"],
    "MipNeRF360 Indoor": ["counter"],
    "Tanks \& Temples": ["truck", "train"],
    "Deep Blending": ["playroom"],
    "LLFF": ["horns", "trex"]
}
# SCENES = ["bycicle", "stump", "counter", "playroom", "truck", "train", "horns", "trex"]

# comparison
comparison = """
    \\begin{tabular}{ccc}
        \\textbf{Ground Truth} & \\textbf{3D-GS} & \\textbf{Ours} \\\\ \\hline"""
for scenes in DIRECTORY.values():
    for scene in scenes:
        comparison += f"""
            \\includegraphics[width=0.26\\textwidth]{{../o-3dgs/eval/{scene}/test/ours_30000/gt/00000.png}} &
            \\includegraphics[width=0.26\\textwidth]{{../o-3dgs/eval/{scene}/test/ours_30000/renders/00000.png}} & 
            \\includegraphics[width=0.26\\textwidth]{{../o-3dgs/eval/{scene}/test/ours_30000/renders/00000.png}} \\\\"""

comparison += """
    \\end{tabular}
    """

with open("report/aux/comparison.tex", "w") as file:
    file.write(comparison)

metrics = """
    \\begin{tabular}{llccccc}
    \\toprule
    \\textbf{Category} & \\textbf{Scene} & \\textbf{Method} & \\textbf{SSIM$\\uparrow$} & \\textbf{PSNR$\\uparrow$} & \\textbf{LPIPS$\\downarrow$} & \\textbf{Mem} \\\\
    \\midrule"""
for cat, scenes in DIRECTORY.items():
    print(cat, scenes)
    if len(scenes) == 0:
        continue
    metrics += f"""
    \\multirow{{{len(scenes) * 2}}}{{*}}{{\\textbf{{{cat}}}}} & """
    for scene in scenes:
        o_3d_gs_metrics = json.load(open(f"o-3dgs/eval/{scene}/results.json"))["ours_30000"]
        our_metrics = json.load(open(f"o-3dgs/eval/{scene}/results.json"))["ours_30000"]
        # our_metrics = json.load(open(f"eval/{scene}/results.json"))["ours_30000"]
        metrics += f"""
        \\multirow{{2}}{{*}}{{{scene.capitalize()}}} 
        & 3D-GS & {o_3d_gs_metrics["SSIM"]:.3f} & {o_3d_gs_metrics["PSNR"]:.2f} & {o_3d_gs_metrics["LPIPS"]:.3f} & {int(o_3d_gs_metrics["Memory"])}MB \\\\"""
        metrics += f"""
        & & \\textbf{{Our Model}} & {our_metrics["SSIM"]:.3f} & {our_metrics["PSNR"]:.2f} & {our_metrics["LPIPS"]:.3f} & {int(our_metrics["Memory"])}MB \\\\"""
        if scene != scenes[-1]:
            metrics += """
            \\cmidrule{2-7} &"""
    if cat != list(DIRECTORY.keys())[-1]:
        metrics += """
        \\midrule"""
metrics += """
    \\bottomrule
    \\end{tabular}"""

with open("report/aux/metrics.tex", "w") as file:
    file.write(metrics)