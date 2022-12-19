#!/usr/bin/env python
# coding: utf-8

# In[1]:


import logging
import os
import sys
import shutil
from glob import glob
from time import time
from tqdm import tqdm
import SimpleITK as sitk
import nibabel as nib
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import monai
from monai.data import list_data_collate, decollate_batch
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.transforms import (
    Activations,
    EnsureChannelFirstd,
    AsDiscrete,
    Compose,
    LoadImaged,
    AddChanneld,
    ScaleIntensityd,
    NormalizeIntensityd,
    Resized,
    RandFlipd,
    ToTensord
)
from monai.visualize import plot_2d_or_3d_image


# In[2]:


ROOT = "Data"
train_image = ROOT+os.sep+"train"+os.sep+"image"
train_mask = ROOT+os.sep+"train"+os.sep+"mask"
test_image = ROOT+os.sep+"test"+os.sep+"image"
test_mask = ROOT+os.sep+"test"+os.sep+"mask"
MODEL_OUTPUT_PATH = "trained_models"
TENSORBOARD_LOGDIR = "logs"
BATCH_SIZE = 4
N_EPOCHS = 100
LR = 7e-4
NUM_WORKERS = 4
N_CLASSES = 1
N_CHANNELS = 1
logging.basicConfig(stream=sys.stdout, level=logging.INFO)


# In[3]:


train_image_paths = sorted(glob(os.path.join(train_image, "*.nii.gz")))
train_mask_paths = sorted(glob(os.path.join(train_mask, "*.nii.gz")))
test_image_paths = sorted(glob(os.path.join(test_image, "*.nii.gz")))
test_mask_paths = sorted(glob(os.path.join(test_mask, "*.nii.gz")))
print(f"Found {len(train_image_paths)} train images and {len(train_mask_paths)} train labels")
print(f"Found {len(test_image_paths)} test images and {len(test_mask_paths)} test labels")

train_files = [{"img": img, "seg": seg} for img, seg in zip(train_image_paths, train_mask_paths)]
val_files = [{"img": img, "seg": seg} for img, seg in zip(test_image_paths, test_mask_paths)]

# define transforms for image and segmentation
train_transforms = Compose(
    [
        LoadImaged(keys=["img", "seg"]),
        EnsureChannelFirstd(keys=["img", "seg"]),
        ScaleIntensityd(keys="img"),
        # NormalizeIntensityd(keys="img"),
        RandFlipd(keys=["img", "seg"]),
        Resized(keys=["img", "seg"], spatial_size=(192, 192, 144), mode="trilinear"),
        ToTensord(keys=["img", "seg"], dtype=torch.float)
    ]
)
val_transforms = Compose(
    [
        LoadImaged(keys=["img", "seg"]),
        EnsureChannelFirstd(keys=["img", "seg"]),
        ScaleIntensityd(keys="img"),
        # NormalizeIntensityd(keys="img"),
        Resized(keys=["img", "seg"], spatial_size=(192, 192, 144), mode="trilinear"),
        ToTensord(keys=["img", "seg"], dtype=torch.float)
    ]
)


# In[4]:


# create a training data loader
train_ds = monai.data.Dataset(data=train_files, transform=train_transforms)
# use batch_size=2 to load images and use RandCropByPosNegLabeld to generate 2 x 4 images for network training
train_loader = DataLoader(
    train_ds,
    batch_size=BATCH_SIZE,
    shuffle=True,
    # num_workers=NUM_WORKERS,
    collate_fn=list_data_collate,
    pin_memory=torch.cuda.is_available(),
)
# create a validation data loader
val_ds = monai.data.Dataset(data=val_files, transform=val_transforms)
val_loader = DataLoader(val_ds,
                        batch_size=BATCH_SIZE,
                        # num_workers=NUM_WORKERS,
                        collate_fn=list_data_collate)


# In[5]:


dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
post_trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])
# create UNet, DiceLoss and Adam optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = monai.networks.nets.UNet(
    spatial_dims=3,
    in_channels=N_CHANNELS,
    out_channels=N_CLASSES,
    channels=(16, 32, 64, 128, 256),
    strides=(2, 2, 2, 2),
    num_res_units=2,
).to(device)
loss_function = monai.losses.DiceLoss(sigmoid=True)
optimizer = torch.optim.Adam(model.parameters(), LR)


# In[6]:


def save_predictions_as_imgs(loader, model, batch_size, epoch, folder="saved_images", device="cuda"):
    model.eval()
    counter = 0
    for idx, data in enumerate(loader):
        x = data["img"].to(device=device)
        y = data["seg"]
        with torch.no_grad():
            prob_map = torch.sigmoid(model(x))
            preds = (prob_map > 0.5).float()
        predictions = preds.detach().cpu().numpy()
        images = x.detach().cpu().numpy()
        probability_maps = prob_map.detach().cpu().numpy()
        for i in range(batch_size):
            try:
                img_sitk = sitk.GetImageFromArray(images[i].squeeze())
                prob_map_sitk = sitk.GetImageFromArray(probability_maps[i].squeeze())
                pred_sitk = sitk.GetImageFromArray(predictions[i].squeeze())
                gt_sitk = sitk.GetImageFromArray(y[i].squeeze())
                # sitk.WriteImage(pred_sitk, f"{folder}/pred_epoch_{epoch}_batch_{idx}_{i}.nii.gz")
                # sitk.WriteImage(gt_sitk, f"{folder}/gt_epoch_{epoch}_batch_{idx}_{i}.nii.gz")
                # sitk.WriteImage(img_sitk, f"{folder}/img_epoch_{epoch}_batch_{idx}_{i}.nii.gz")
                # sitk.WriteImage(prob_map_sitk, f"{folder}/probability_map_epoch_{epoch}_batch_{idx}_{i}.nii.gz")
                sitk.WriteImage(pred_sitk, f"{folder}/pred_epoch_{epoch}_{counter}.nii.gz")
                sitk.WriteImage(gt_sitk, f"{folder}/gt_epoch_{epoch}_{counter}.nii.gz")
                sitk.WriteImage(img_sitk, f"{folder}/img_epoch_{epoch}_{counter}.nii.gz")
                sitk.WriteImage(prob_map_sitk, f"{folder}/probability_map_epoch_{epoch}_{counter}.nii.gz")
                counter += 1
            except IndexError:
                pass
        # torchvision.utils.save_image(x.float()[:32], f"{folder}/img_{idx}.png")
        # torchvision.utils.save_image(preds[:32], f"{folder}/pred_{idx}.png")
        # torchvision.utils.save_image(y[:32], f"{folder}/gt_{idx}.png")
    model.train()

def clear_directory(directory_path):
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

# In[8]:


# start a typical PyTorch training
val_interval = 2
best_metric = -1
best_metric_epoch = -1
epoch_train_loss_values = list()
epoch_val_loss_values = list()
epoch_train_dice_values = list()
epoch_val_dice_values = list()
metric_values = list()
clear_directory(TENSORBOARD_LOGDIR)
writer = SummaryWriter(log_dir=TENSORBOARD_LOGDIR)
start_time = time()
for epoch in range(N_EPOCHS):
    print("-" * 10)
    print(f"epoch {epoch + 1}/{N_EPOCHS}")
    model.train()
    epoch_loss = 0
    step = 0
    epoch_len = len(train_ds) // train_loader.batch_size
    for batch_data in tqdm(train_loader):
        step += 1
        inputs, labels = batch_data['img'].to(device), batch_data['seg'].to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        # outputs = outputs.squeeze(dim=1)
        preds = post_trans(outputs)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        # print(f"{step}/{epoch_len}, train_loss: {loss.item():.4f}")
        dice_metric(y_pred=preds.unsqueeze(dim=1), y=labels.unsqueeze(dim=1))
    # aggregate the final mean dice result
    train_epoch_dice = dice_metric.aggregate().item()
    # reset the status for next validation round
    dice_metric.reset()
    # train_dice = np.asarray(train_dice_values).mean()
    writer.add_scalars("Train_metrics", {"Loss": loss.item(), "Dice Score": train_epoch_dice}, epoch+1)
    epoch_loss /= step
    epoch_train_loss_values.append(epoch_loss)
    epoch_train_dice_values.append(train_epoch_dice)
    print(f"Epoch {epoch + 1}\tAverage loss: {epoch_loss:.4f}\tAverage Dice: {train_epoch_dice:.4f}")

    # Validation
    epoch_val_loss = 0
    step = 0
    model.eval()
    with torch.no_grad():
        for val_data in tqdm(val_loader):
            step += 1
            val_images, val_labels = val_data['img'].to(device), val_data['seg'].to(device)
            val_outputs = model(val_images)
            val_preds = post_trans(val_outputs)
            # print("Validation output data voxel values")
            # print(torch.unique(val_outputs))
            # compute metric for current iteration
            loss = loss_function(val_outputs, val_labels)
            epoch_val_loss += loss.item()
            dice_metric(y_pred=val_preds.unsqueeze(dim=1), y=val_labels.unsqueeze(dim=1))
    # aggregate the final mean dice result
    val_epoch_dice = dice_metric.aggregate().item()
    # reset the status for next validation round
    dice_metric.reset()

    epoch_val_loss /= step
    epoch_val_loss_values.append(epoch_val_loss)
    epoch_val_dice_values.append(val_epoch_dice)
    if val_epoch_dice > best_metric:
        best_metric = val_epoch_dice
        best_metric_epoch = epoch + 1
        torch.save(model.state_dict(), os.path.join(MODEL_OUTPUT_PATH, "best_metric_model_segmentation3d_dict.pth"))
        print("Saved new best metric model")
        # save_predictions_as_imgs(val_loader, model, BATCH_SIZE, epoch)
    # print(
    #     "Epoch: {}\tValidation Loss: {:.4f}\tCurrent mean dice: {:.4f} Best mean dice: {:.4f} at epoch {}".format(
    #         epoch + 1, epoch_val_loss, val_epoch_dice, best_metric, best_metric_epoch
    #     )
    # )
    writer.add_scalars("Validation metrics", {"Loss": epoch_val_loss, "Dice": val_epoch_dice}, epoch + 1)
    # plot the last model output as GIF image in TensorBoard with the corresponding image and label
    # plot_2d_or_3d_image(val_images, epoch + 1, writer, index=0, tag="image")
    # plot_2d_or_3d_image(val_labels, epoch + 1, writer, index=0, tag="label")
    # plot_2d_or_3d_image(val_outputs, epoch + 1, writer, index=0, tag="output")

    # if epoch > 0:
    #     save_predictions_as_imgs(val_loader, model, BATCH_SIZE, epoch)

end_time = time()
time_taken = (end_time - start_time)/60
print(f"Training completed in {time_taken} minutes, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}")
writer.close()


# In[ ]:




