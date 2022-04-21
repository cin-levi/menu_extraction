import json
import shutil, os
from glob import glob
from tqdm import tqdm

# full_data_folder = 'D:\\menu_extraction\\data\\pdf'
# test_data_folder = 'D:\\menu_extraction\\data\\pdf_test'
#
# all_data_files = glob(full_data_folder + '/*.pdf')
# test_data_names = [os.path.basename(x) for x in glob(test_data_folder + '/*.pdf')]
#
# with open('D:\\menu_extraction\\utils/test_list.json', 'w', encoding='utf-8') as f:
#     json.dump(test_data_names, f, ensure_ascii=False)
#
# all_training_files = [x for x in all_data_files if os.path.basename(x) not in test_data_names]
#
# print("Len training: ", len(all_training_files))
# print("Len testing: ", len(test_data_names))
#
# train_data_folder = 'D:\\menu_extraction\\data\\pdf_train'
#
# for file in tqdm(all_training_files):
#     dst = train_data_folder + f'/{os.path.basename(file)}'
#     shutil.copy(file, dst)

files = glob('D:\\menu_extraction\\data\\json\\full/*json')

import random

test_samples = random.sample(files, 40)
train_samples = [x for x in files if x not in test_samples]

train_data_folder = 'D:\\menu_extraction\\data\\json\\train'
for file in tqdm(train_samples):
    dst = train_data_folder + f'/{os.path.basename(file)}'
    shutil.copy(file, dst)

test_data_folder = 'D:\\menu_extraction\\data\\json\\test'
for file in tqdm(test_samples):
    dst = test_data_folder + f'/{os.path.basename(file)}'
    shutil.copy(file, dst)
