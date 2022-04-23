import json
from glob import glob
import shutil
import os
from tqdm import tqdm

def get_testing_list():
    testing_folder = 'D:\\menu_extraction\\data\\json\\test'
    testing_files = glob(testing_folder + '/*.json')
    pdf_files = [os.path.basename(file)[:-5] for file in testing_files]  # .json
    print(len(pdf_files))
    testing_list = 'D:\\menu_extraction\\utils\\test_list.json'
    with open(testing_list, 'w', encoding='utf-8') as f:
        json.dump(pdf_files, f, ensure_ascii=False)


def split_training_testing_files(testing_list, pdf_folder, training_folder, testing_folder):
    files = glob(pdf_folder + '/*.pdf')
    testing_files = [file for file in files if os.path.basename(file) in testing_list]
    assert len(testing_files) == len(testing_list)
    training_files = [f for f in files if f not in testing_files]
    for f in tqdm(testing_files):
        output_file = testing_folder + '/'+ os.path.basename(f)
        shutil.copy(f, output_file)
    for f in tqdm(training_files):
        output_file = training_folder + '/'+ os.path.basename(f)
        shutil.copy(f, output_file)
    pass

if __name__ == '__main__':
    # get_testing_list()
    testing_list = json.load(open('D:\\menu_extraction\\utils\\test_list.json', 'r', encoding='utf-8'))
    pdf_folder = 'D:\\menu_extraction\\data\\pdf'
    training_folder = 'D:\\menu_extraction\\data\\pdf_train'
    testing_folder = 'D:\\menu_extraction\\data\\pdf_test'
    split_training_testing_files(testing_list, pdf_folder, training_folder, testing_folder)
    pass
