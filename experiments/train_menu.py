from models.trainer import Trainer
import json

from processors import prj_path

# from neptune_credentials import fastqa_credential
import os

fold_prefix = 'fold_'


def train_kfold(data_folder, output_folder):
    all_reports = []
    for i in range(3):
        data_dir = data_folder + f'/{fold_prefix}{i}/'
        train_file = f'train.json'
        test_file = 'test.json'
        label_dict_file = data_dir + 'question_list.json'  # can be list or dict

        output_dir = output_folder + f'/{fold_prefix}{i}'
        cache_dir = prj_path + f'/caches/{fold_prefix}{i}'

        all_labels = sorted([x for x in json.load(open(label_dict_file, 'r', encoding='utf-8'))])

        attention_type = 'Single'

        tokenizer_name = 'microsoft/layoutlm-base-uncased'
        pretrained_path = 'microsoft/layoutlm-base-uncased'

        trainer = Trainer(pretrained_path, tokenizer_name, all_labels, embedding_from_encoder=False,
                          use_multiple_attention=attention_type == 'Multiple',
                          max_seq_len=128, doc_stride=32, version='v1')

        training_mode = 'QANER'
        batch_size = 8

        report_fold = trainer.train(data_dir=data_dir, train_filename=train_file, test_filename=test_file,
                                    output_dir=output_dir,
                                    train_bs=batch_size, test_bs=batch_size, num_train_epochs=10, cache_dir=cache_dir,
                                    override_cache=True,
                                    report_name=f'menu_v1_{fold_prefix}{i}' + training_mode,
                                    training_mode=training_mode,
                                    neptune_credential=fastqa_credential
                                    )
        all_reports.append(report_fold)
    with open(output_folder + '/full_report.json', 'w', encoding='utf-8') as f:
        json.dump(all_reports, f, ensure_ascii=False)


def train_full_data(data_dir, output_folder):
    train_file = f'train.json'
    test_file = 'test.json'
    label_dict_file = data_dir + 'question_list.json'  # can be list or dict

    output_dir = output_folder
    cache_dir = prj_path + f'/caches'

    all_labels = sorted([x for x in json.load(open(label_dict_file, 'r', encoding='utf-8'))])

    attention_type = 'Single'

    tokenizer_name = 'microsoft/layoutlm-base-uncased'
    pretrained_path = 'microsoft/layoutlm-base-uncased'

    trainer = Trainer(pretrained_path, tokenizer_name, all_labels, embedding_from_encoder=False,
                      use_multiple_attention=attention_type == 'Multiple',
                      max_seq_len=128, doc_stride=32, version='v1')

    training_mode = 'QANER'
    batch_size = 8

    trainer.train(data_dir=data_dir, train_filename=train_file, test_filename=test_file,
                  output_dir=output_dir,
                  train_bs=batch_size, test_bs=batch_size, num_train_epochs=10, cache_dir=cache_dir,
                  override_cache=True,
                  report_name=f'menu_v1_test_stable' + training_mode,
                  training_mode=training_mode,
                  # neptune_credential=fastqa_credential
                  )


if __name__ == '__main__':
    data_folder = prj_path + '/data/'  # /kfold'
    output_folder = prj_path + '/outputs'
    # train_kfold(data_folder, output_folder)
    train_full_data(data_folder, output_folder)
