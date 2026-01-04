import numpy as np
import torch
import argparse
import cv2
import torch.utils.data as data
import torchvision.transforms as transforms
import random
from PIL import Image
from PIL import ImageFile
import os
import re

def default_toTensor(img):
    t_list = [transforms.ToTensor()]
    composed_transform = transforms.Compose(t_list)
    return composed_transform(img)

def create_dataset(
    args,
    data_path,
    mode='train',
):
    def _list_image_files_recursively(data_dir):
        file_list = []
        root  =''
        for home, dirs, files in os.walk(data_dir):
            for filename in files:
                ext = filename.split(".")[-1]
                if ext.lower() in ["jpg", "jpeg", "png", "gif", "webp","bmp"]:
                    file_list.append(filename)
        root = home
        def sort_key(filename):
            nums = [int(num) for num in re.findall(r'\d+', filename)]

            return (nums[0], nums[1], nums[3], nums[2])

        sorted_filenames = sorted(file_list, key=sort_key)
        sorted_filenames = [root + item for item in sorted_filenames ]

        return sorted_filenames
    if mode != 'demo_test':
        data_path = data_path +'sample/'
    else:
        data_path = data_path +'/'

        args.BATCH_SIZE = 1

    uhdm_files = _list_image_files_recursively(data_dir=data_path)
    dataset = uhdm_data_loader(args, uhdm_files, mode=mode)

    data_loader = data.DataLoader(
        dataset, batch_size=args.BATCH_SIZE, shuffle=False, num_workers=args.WORKER, drop_last=True
    )

    return data_loader

class uhdm_data_loader(data.Dataset):

    def __init__(self, args, image_list, mode='train'):
        self.image_list = image_list
        self.args = args
        self.EXP_NAME = args.EXP_NAME
        self.channel =args.CHANNEL
    def __getitem__(self, index):

        data ={}

        images=[]
        images_path =[]
        start_idx = index * 8

        for i in range(start_idx, start_idx + 8):
            img_name = self.image_list[i]
            split = os.path.basename(img_name).split("_")[-3]

            if self.EXP_NAME =='result_1' and split =='1' :
                image = Image.open(img_name).convert('L')
                images.append(image)
                images_path.append(img_name)

            if self.EXP_NAME =='result_4' and (split =='1'or split =='2'or split =='3'or split =='4') :
                image = Image.open(img_name).convert('L')
                images.append(image)
                images_path.append(img_name)

            if self.EXP_NAME !='result_1' and self.EXP_NAME !='result_4' :
                if split =='1' or split =='2' or split =='3' or split =='4':
                    image = Image.open(img_name).convert('L')
                    images.append(image)
                    images_path.append(img_name)

                if split == '9' and self.EXP_NAME == 'result_s5_9' :  # s2
                    image = Image.open(img_name).convert('L')
                    images.append(image)
                    images_path.append(img_name)
                elif split == '8' and self.EXP_NAME == 'result_AOP_8':  # AOP
                    image = Image.open(img_name).convert('L')
                    images_path.append(img_name)
                    images.append(image)

                elif split == '7' and self.EXP_NAME == 'result_DOLP_7':
                    image = Image.open(img_name).convert('L')
                    images_path.append(img_name)
                    images.append(image)

                elif self.EXP_NAME == 'result_s2_6' and split == '6' :
                    image = Image.open(img_name).convert('L')
                    images_path.append(img_name)
                    images.append(image)

                elif self.EXP_NAME == 'result_s5_AOP_8_9' and (split=='9' or split =='8'):
                    image = Image.open(img_name).convert('L')
                    images_path.append(img_name)
                    images.append(image)

                elif self.EXP_NAME == 'result_AOP_DOLP_7_8' and (split=='7' or split =='8'):
                    image = Image.open(img_name).convert('L')
                    images_path.append(img_name)
                    images.append(image)

                elif self.EXP_NAME == 'result_DOLP_s5_7_9' and (split=='9' or split =='7'):
                    image = Image.open(img_name).convert('L')
                    images_path.append(img_name)
                    images.append(image)

                elif self.EXP_NAME == 'result_s2_DOLP_6_7' and (split=='6' or  split =='7'):
                    image = Image.open(img_name).convert('L')
                    images_path.append(img_name)
                    images.append(image)


                elif self.EXP_NAME == 'result_s2_AOP_DOLP_6_8_9' and (split=='6' or split =='7'or split =='8'):
                    image = Image.open(img_name).convert('L')
                    images_path.append(img_name)
                    images.append(image)


        sample_input = np.stack(images, axis=-1)

        sample= default_toTensor(sample_input)

        filename = os.path.splitext(self.image_list[start_idx])[0]

        numbers = [int(num) for num in re.findall(r'\d+', filename)]

        label_name = f"{numbers[0]}_{numbers[-1]}_gt.bmp"

        label_path = self.args.TRAIN_DATASET+ 'label/'+label_name

        img=Image.open(label_path).convert('L')
        label = default_toTensor(img)

        data['in_img'] = sample
        data['label'] = label
        data['number'] = numbers[-1]
        return data

    def __len__(self):
        return len(self.image_list) // 8  



