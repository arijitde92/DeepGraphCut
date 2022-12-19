import imcut.pycut
import numpy as np
import SimpleITK as sitk
import torch
import warnings
import os
from time import time
from glob import glob
from tqdm import tqdm
from monai.metrics import DiceMetric
warnings.filterwarnings("ignore")

ROOT_PATH = "saved_images"
original_images = glob(ROOT_PATH + os.sep + "img_*.nii.gz")
gt_images = glob(ROOT_PATH + os.sep + "gt_*.nii.gz")
pred_images = glob(ROOT_PATH + os.sep + "pred_*.nii.gz")
probability_maps = glob(ROOT_PATH + os.sep + "probability_map_*.nii.gz")

assert len(original_images) == len(gt_images) == len(pred_images) == len(probability_maps), "Inconsistent number of images found"
print(f"Found {len(gt_images)} ground truths, original images, probability maps and Unet predicted segmentation maps")

def min_max_scale(img_path):
    img = sitk.GetArrayFromImage(sitk.ReadImage(img_path))
    min_val = img.min()
    max_val = img.max()
    img = (img - min_val)/(max_val - min_val)
    return img


dice_metric = DiceMetric(include_background=True, reduction='mean', get_not_nans=False)
start_time = time()
for i in tqdm(range(len(original_images))):
    img = min_max_scale(original_images[i])
    seed = sitk.GetArrayFromImage(sitk.ReadImage(pred_images[i]))
    prob_map = sitk.GetArrayFromImage(sitk.ReadImage(probability_maps[i]))
    prob_map = np.where(prob_map > 0.9, 0.9, 0.1)
    gt = sitk.GetArrayFromImage(sitk.ReadImage(gt_images[i]))
    gt = np.where(gt > 0.5, 1.0, gt)
    seed = np.where(seed == 0, 2, seed)
    # print("\nSeed: ", np.unique(seed, return_counts=True))
    # print("Image: ", np.unique(img, return_counts=True))
    # print("Prob map:", np.unique(prob_map, return_counts=True))
    # print("Image Shape:", img.shape, "\tSeed shape:", seed.shape, "\tProbability map shape:", prob_map.shape, "\tGround truth shape:", gt.shape)
    # print("Image dtype:", img.dtype, "\tSeed dtype:", seed.dtype, "\tProbability map dtype:", prob_map.dtype, "\tGround truth dtype:", gt.dtype)
    # print(f"Processing data number {i}")
    gc = imcut.pycut.ImageGraphCut(img, prob_map)
    gc.set_seeds(seed)
    gc.run()
    segmentation = np.where(gc.segmentation == 0, 1, 0)
    # print("GraphCut output", np.unique(segmentation), segmentation.shape)
    sitk.WriteImage(sitk.GetImageFromArray(segmentation), f"DGC_output/dgc_output_{i}.nii.gz")
    dgc_pred = torch.from_numpy(segmentation).unsqueeze(dim=0).unsqueeze(dim=0)
    ground_truth = torch.from_numpy(gt).unsqueeze(dim=0).unsqueeze(dim=0)
    dice_metric(y_pred=dgc_pred, y=ground_truth)
end_time = time()
test_dice = dice_metric.aggregate().item()
print("Average Dice Score:", test_dice)
print("Total time taken: {} minutes".format((end_time - start_time)/60))

