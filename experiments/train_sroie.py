from models.trainer import Trainer
import json

from processors import prj_path

fastqa_credential = None
import os

if __name__ == '__main__':
    tokenizer_name = 'microsoft/layoutlm-base-uncased'
    pretrained_path = 'microsoft/layoutlm-base-uncased'

    data_dir = prj_path + '/data/sroie/'
    train_file = f'train.json'
    test_file = 'test.json'
    output_dir = prj_path + '/outputs'
    cache_dir = prj_path + '/caches'

    label_dict_file = data_dir + 'question_list.json'  # can be list or dict

    all_labels = sorted([x for x in json.load(open(label_dict_file, 'r', encoding='utf-8'))])

    attention_type = 'Multiple'
    trainer = Trainer(pretrained_path, tokenizer_name, all_labels, embedding_from_encoder=False,
                      use_multiple_attention=attention_type == 'Multiple',
                      max_seq_len=256, doc_stride=64)

    training_mode = 'NER'
    batch_size = 2

    trainer.train(data_dir=data_dir, train_filename=train_file, test_filename=test_file, output_dir=output_dir,
                  train_bs=batch_size, test_bs=batch_size, num_train_epochs=30, cache_dir=cache_dir,
                  override_cache=True,
                  report_name='cord_layoutlm_' + training_mode,
                  training_mode=training_mode,
                  neptune_credential=fastqa_credential
                  )  # training_mode in ['FastQA', 'NER', 'FastQANER']
