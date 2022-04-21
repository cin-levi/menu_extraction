import json
import os.path

result_file = 'D:\\menu_extraction\\evaluate_detailed.json'

with open(result_file, 'r', encoding='utf-8') as f:
    result = json.load(f)

worst = []  # < 0.4

bad = []  # < 0.7

for file in result:
    if result[file]['w']['f'] < 0.5:
        worst.append(file)
    elif result[file]['w']['f'] < 0.7:
        bad.append(file)

print(len(result))
print(len(worst))
print(len(bad))
with open('D:\\menu_extraction\\file_examination/worst.json', 'w', encoding='utf-8') as f:
    json.dump(worst, f, ensure_ascii=False)

with open('D:\\menu_extraction\\file_examination/bad.json', 'w', encoding='utf-8') as f:
    json.dump(bad, f, ensure_ascii=False)


from glob import glob

testing_folder = 'D:\\menu_extraction\\data\\json\\test'
testing_files = glob(testing_folder+'/*.json')
accepted_testing_files = [file for file in testing_files if os.path.basename(file) not in worst]
print(len(accepted_testing_files))

from tqdm import tqdm
import shutil
# Remove testing file
for file in tqdm(accepted_testing_files):
    dst = f'D:\\menu_extraction\\data\\json_filtered\\test/{os.path.basename(file)}'
    shutil.copy(file, dst)


training_folder = 'D:\\menu_extraction\\data\\json\\train'
training_files = glob(training_folder+'/*.json')
accepted_training_files = [file for file in training_files if os.path.basename(file) not in worst]
print(len(accepted_training_files))

for file in tqdm(accepted_training_files):
    dst = f'D:\\menu_extraction\\data\\json_filtered\\train/{os.path.basename(file)}'
    shutil.copy(file, dst)
