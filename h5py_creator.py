
import h5py
import torch
from data import load_data, data_set
from tqdm import tqdm
import numpy as np


# hf = h5py.File('data.h5', 'r')

# print(np.array(hf.get('S1/Directions-1/imageSequence/54138969/img_001370.jpg')))

# exit()


hf = h5py.File('data.h5', 'w')

for dataset in ["train", "validation"]:

    data = data_set(dataset)

    loader = torch.utils.data.DataLoader(
        data, batch_size=1, num_workers=10, pin_memory=True, shuffle=True, drop_last=False)

    iterator = iter(loader)

    # print(batch["image_name"][0].split("/")[-5:])

    for iteration in tqdm(range(len(loader))):
        batch = next(iterator)

        image_names = batch["image_name"][0].split("/")[-5:]
        mask_names = batch["mask_name"][0].split("/")[-5:]

        image_name = f"{image_names[0]}/{image_names[1]}/{image_names[2]}/{image_names[3]}/{image_names[4]}"
        mask_name = f"{mask_names[0]}/{mask_names[1]}/{mask_names[2]}/{mask_names[3]}/{mask_names[4]}"

        hf.create_dataset(image_name, data=batch["image"][0], compression="gzip", compression_opts=9)
        hf.create_dataset(mask_name, data=batch["mask_rcnn"][0], compression="gzip", compression_opts=9)

hf.close()
