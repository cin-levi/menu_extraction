import json
import random

fold_prefix = 'fold_'
data_folder = 'D:\\menu_extraction\\data\\kfold/'
data_file = 'D:\\menu_extraction\\data\\full.json'

with open(data_file, 'r', encoding='utf-8') as f:
    data = json.load(f)['data']
print(len(data))

k = 3
avg_num_each_fold = int(len(data) / k)
all_indexes = range(len(data))

fold_0_indexes = random.sample(all_indexes, avg_num_each_fold)
remaining_indexes = [x for x in all_indexes if x not in fold_0_indexes]
fold_1_indexes = random.sample(remaining_indexes, avg_num_each_fold)
fold_2_indexes = [x for x in remaining_indexes if x not in fold_1_indexes]

fold_indexes = [fold_0_indexes, fold_1_indexes, fold_2_indexes]

for i in range(k):
    output_folder = data_folder + f'{fold_prefix}{i}'
    # test_idx = fold_indexes[i]
    test_idx = all_indexes[i * avg_num_each_fold: (i + 1) * avg_num_each_fold]
    train_idx = [idx for idx in all_indexes if idx not in test_idx]
    train_samples = [data[idx] for idx in train_idx]
    test_samples = [data[idx] for idx in test_idx]
    with open(output_folder + '/train.json', 'w', encoding='utf-8') as f:
        json.dump({'data': train_samples}, f, ensure_ascii=False)
    with open(output_folder + '/test.json', 'w', encoding='utf-8') as f:
        json.dump({'data': test_samples}, f, ensure_ascii=False)
