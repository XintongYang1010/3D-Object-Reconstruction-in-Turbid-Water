
import os
import shutil
from PIL import Image
from PIL import ImageFile
import numpy as np
import cv2
import re
def sampling(root_dir ,target_dir):
    num1 = 0
    num2 = 0
    bf_parent_dirname_2 = ""
    bf_parent_dirname_1 = ""

    for dirpath, dirnames, filenames in os.walk(root_dir):

        parent_dirname_2 = os.path.basename(os.path.dirname(os.path.dirname(dirpath)))

        parent_dirname_1 = os.path.basename(os.path.dirname(dirpath))

        current_dirname = os.path.basename(dirpath)

        for filename in filenames:
            if filename.endswith('.bmp'):
                # if (bf_parent_dirname_2 != parent_dirname_2):
                #     # num2 = num2 + 1
                #     # num2=parent_dirname_2.split('_')[-1]
                #     # print(parent_dirname_2)
                #     bf_parent_dirname_2 = parent_dirname_2
                if (bf_parent_dirname_1 != parent_dirname_1):
                    # num1 = num1 + 1
                    num1 = parent_dirname_1.split('_')
                    bf_parent_dirname_1 = parent_dirname_1

                new_filename = f"{num1[-2]}_{num1[-1]}_{current_dirname}_{os.path.splitext(os.path.basename(filename))[0]}_moire.bmp"

                old_filepath = os.path.join(dirpath, filename)

                new_filepath = os.path.join(target_dir, new_filename)

                shutil.copyfile(old_filepath, new_filepath)



    print("===================get feature===================")
    get_feature(target_dir,target_dir)
def get_feature(data_dir,target_path):

    def _list_image_files_recursively(data_dir):
        filename_list = []
        file_home=''
        for home, dirs, files in os.walk(data_dir):
            for filename in files:
                ext = filename.split(".")[-1]
                if ext.lower() in ["jpg", "jpeg", "png", "gif", "webp","bmp"]:
                    file_home=home
                    filename_list.append(filename)

        def sort_key(filename):
            nums = [int(num) for num in re.findall(r'\d+', filename)]

            return (nums[0], nums[1], nums[3], nums[2])

        sorted_filenames = sorted(filename_list, key=sort_key)

        file_list = [file_home +"/"+ element for element in sorted_filenames]

        return file_list

    # 90->1  45->2  135->3  0->4
    # S0 = (I0+I45+I90+I135)/2
    # S0 = (4+2+1+3)/2

    # S1 = (I0 - I90) / S0
    # S2 = (I45 - I135) / S0
    # DoLP = sqr(S1 * S1 + S2 * S2) / S0
    # AoP = arctan(S2 / S1) / 2


    # s1 =(4-1)/s0
    # s2 = (2-3)/s0

    # s5 = s1/s2
    # DOLP = sqr(S1 * S1 + S2 * S2) / S0
    # AOP = arctan(S2 / S1) / 2
    # DOP = ( S1^2+S2^2+S3^2)**(1/2)/S0

    def cal_feature(I90,I45,I0,I135):

        def normal(data):

            min_value = np.min(data)
            max_value = np.max(data)

            s0 = (data - min_value) / (max_value - min_value)

            return s0

        s0 = I0 + I90
        s0 = np.where(s0== 0, 1, s0)

        s1_1 =(I0 - I90)  *255
        s1_2 =normal(I0 - I90)
        s1 = np.clip(s1_1, 0, 255)

        s2_1 =  (I45 - I135) *255
        s2_2 = normal(I45 - I135)
        s2 = np.clip(s2_1, 0, 255)

        DoLP = np.sqrt(s1_1 ** 2 + s2_1 ** 2) / s0 *255

        DoLP= np.clip(DoLP, 0, 255)

        s1_2 = np.where(s1_2 == 0, 1, s1_1)

        Aop = np.arctan2(s2_2, s1_2) * 0.5 * 70
        Aop= np.clip(Aop, 0, 255)

        # Dop = (s1_1**2 + s2_1**2 + S3**2)**(0.5) / s0
        s2_3 =  np.where(s2_1 == 0, 1, s1_1)
        s5 = s1_1/s2_3
        s5 = np.clip(s5, 0, 255)

        dict = {
                # '5': s1,
                # '6': s2,
                '7': DoLP,
                '8': Aop,
                '9': s5
                }

        return  dict
    data = _list_image_files_recursively(data_dir)

    data_length = len(data)//4


    # 代替 __getitem
    for  i in range(data_length) :
        images = []

        i = i * 4
        for j in range(i,i+4):
            img_name = data[j]
            image = Image.open(img_name).convert('L')
            images.append(np.array(image).astype(np.float64))

        dict = cal_feature(images[0],images[1],images[2],images[3])

        filename = os.path.splitext(os.path.basename(data[i]))[0]

        numbers = [int(num) for num in re.findall(r'\d+', filename )]

        for key, value in dict.items():
            feature_path = target_path+f'/{numbers[0]}_{numbers[1]}_{key}_{numbers[-1]}_moire.bmp'
            cv2.imwrite(feature_path, value)


def get_label(root_dir,destination_folder):
    num=0
    bf_parent_dirname = ""

    for root, dirs, files in os.walk(root_dir):

        current_dirname = os.path.basename( root)
        for file in files:

            if file.endswith('.bmp'):

                if (bf_parent_dirname != current_dirname):
                    num = current_dirname.split('_')[-1]
                    bf_parent_dirname = current_dirname

                new_filename = f"{num}_{os.path.splitext(os.path.basename(file))[0]}_gt.bmp"

                source_file_path = os.path.join(root, file)
                destination_file_path = os.path.join(destination_folder, new_filename)


                shutil.copyfile(source_file_path, destination_file_path)


def train_val(split):
    if split=='train':
        root_dir_sample = ''
        target_dir_sample = ''

        root_dir_label = ''
        target_dir_label = ''

        files = [target_dir_sample,target_dir_label]
        for f in files:
            if not os.path.exists(f):
                os.makedirs(f)

        sampling (root_dir_sample, target_dir_sample)
        get_label(root_dir_label,target_dir_label)
    elif split=='test':

        root_dir_sample = ''
        target_dir_sample = ''

        root_dir_label = ''
        target_dir_label = ''

        files = [target_dir_sample,target_dir_label]
        for f in files:
            if not os.path.exists(f):
                os.makedirs(f)

        sampling (root_dir_sample, target_dir_sample)
        get_label(root_dir_label,target_dir_label)


if __name__=='__main__':

    train_val('train')

    train_val('test')