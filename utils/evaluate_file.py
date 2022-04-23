import os, json, datetime
import numpy as np
import torch
from tqdm import tqdm

from models.layoutlm_model import NERLayoutLM
# from models.layoutlmv2_model import NERLayoutLMv2
from models.optimization import AdamW, get_linear_schedule_with_warmup
from processors.fastqa_based_processor import CustomProcessor, Result, NerResult

from transformers import AutoTokenizer
# from layoutlmv2.tokenization_layoutlmv2 import LayoutLMv2Tokenizer
from torch.utils.data import DataLoader

from processors import prj_path

cuda = torch.cuda.is_available()

SAVE_EVERY = 5


def dump_report(ner_predictions, test_examples):
    detailed_predictions = []
    for i, example in enumerate(test_examples):
        uid = example.uid
        question_texts = example.question_texts
        answers = example.answers
        qa = {question_texts[i]: answers[i] for i in range(len(answers))}
        predictions = ner_predictions[uid]
        ca = qa['w']
        pred = [x for x in predictions if x['entity_type'] == 'w']
        ca = [x['text'] for x in ca]
        pred = [x['text'] for x in pred]
        detailed_predictions.append({"ground_truth": ca, "predictions": pred})
    return detailed_predictions


class Evaluator(object):
    def __init__(self, pretrained_path, tokenizer_name, all_labels, max_seq_len=384, doc_stride=128, max_ans_len=60,
                 embedding_from_encoder=False, use_multiple_attention=False, version='v1'):
        if version == 'v1':
            self.model = NERLayoutLM.from_pretrained(pretrained_path, num_question=len(all_labels),
                                                     max_seq_len=max_seq_len, init_embedding=None,
                                                     use_multiple_attention=use_multiple_attention,
                                                     embedding_from_encoder=embedding_from_encoder)
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=False)

        self.processor = CustomProcessor(tokenizer=self.tokenizer,
                                         max_seq_len=max_seq_len,
                                         max_query_len=60,
                                         max_ans_len=max_ans_len,
                                         doc_stride=doc_stride,
                                         question_list=all_labels,
                                         label_list=['0', '1'],  # can be use in classification mode
                                         version_2_with_negative=True)

        self.metadata = {'tokenizer_name': tokenizer_name, 'question_list': all_labels, 'max_seq_len': max_seq_len,
                         'doc_stride': doc_stride, 'max_ans_len': max_ans_len,
                         'use_multiple_attention': use_multiple_attention}

        if cuda:
            self.model.cuda()

    def calculate_accuracy(self, logits, target):
        pass

    def load_dataset(self, data_path):
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)['data']
        examples = {}
        print("get examples")
        for i, document in tqdm(enumerate(data)):
            title = document.get('title', "")
            if title not in examples:
                examples[title] = {'examples': []}
            for a_paragraph in document['paragraphs']:
                examples[title]['examples'].append(self.processor.read_qa_examples(a_paragraph, title))
        print("convert example to features")
        for title in examples:
            features = self.processor.convert_examples_to_features(examples[title]['examples'])
            dataset = self.processor.features_to_dataset(features)
            examples[title]['features'] = features
            examples[title]['dataset'] = dataset
        return examples

    # calculate by position
    def evaluate(self, test_dataloader, test_features, test_examples, epoch, training_mode,
                 evaluation_function=None, ca_data=None):
        # training_mode in ['FastQA', 'NER', 'FastQANER']
        self.model.eval()  # set model to eval mode

        all_results = []
        ner_results = []

        for batch in tqdm(test_dataloader):
            # input data part parameter
            input_ids = batch[0].cuda() if cuda else batch[0]
            attention_mask = batch[1].cuda() if cuda else batch[1]
            token_type_ids = batch[2].cuda() if cuda else batch[2]
            bbox = batch[3].cuda() if cuda else batch[3]
            # label part
            sequence_label_id = batch[-6].cuda() if cuda else batch[-6]
            token_label_ids = batch[-5].cuda() if cuda else batch[-5]
            question_indexes = batch[-4].cuda() if cuda else batch[-4]
            start_positions = batch[-3].cuda() if cuda else batch[-3]
            end_positions = batch[-2].cuda() if cuda else batch[-2]
            feature_indices = batch[-1]
            return_dict = False
            outputs = None  # clean buffer memory

            outputs = self.model(input_ids=input_ids,
                                 attention_mask=attention_mask,
                                 token_type_ids=token_type_ids,
                                 bbox=bbox,
                                 sequence_label_id=sequence_label_id,
                                 token_label_ids=token_label_ids,
                                 question_indexes=question_indexes,
                                 start_positions=start_positions,
                                 end_positions=end_positions,
                                 return_dict=return_dict)
            start_logits = outputs[2].detach().cpu()
            end_logits = outputs[3].detach().cpu()
            ner_logits = outputs[4].detach().cpu()

            for i, feature_index in enumerate(feature_indices):
                feature = test_features[feature_index.item()]
                starts = start_logits[i]
                ends = end_logits[i]
                logits = ner_logits[i]

                q_ids = feature.q_ids

                for j in range(len(q_ids)):
                    result_id = feature.uid + ' ' + q_ids[j]
                    all_results.append(
                        Result(uid=result_id, start_logits=starts[j].numpy(), end_logits=ends[j].numpy()))

                ner_results.append(NerResult(uid=feature.uid, logits=logits, gt=feature.token_label_ids))

        predictions, nbest_predictions = self.processor.compute_predictions(
            test_examples, test_features, all_results, None)

        ner_predictions = self.processor.compute_ner_predictions(test_examples, test_features, ner_results)

        # todo: Combine NER and QA here
        if training_mode == 'QANER':
            ner_predictions = self.processor.combine_ner_and_QA_result(ner_predictions, nbest_predictions,
                                                                       test_examples)
        elif training_mode == 'QA':
            ner_predictions = {uid: [] for uid in ner_predictions}  # remove result of NER
            ner_predictions = self.processor.combine_ner_and_QA_result(ner_predictions, nbest_predictions,
                                                                       test_examples)

        #
        detailed_predictions = dump_report(ner_predictions, test_examples)
        (macro_ner, micro_ner, ner_detail) = self.processor.ner_evaluate_by_field(ner_predictions, test_examples)

        nbest_to_save = {id: [{'text': x['text'], 'probability': x['probability']} for x in nbest_predictions[id]] for
                         id in nbest_predictions}

        output_path = prj_path + '/outputs'
        with open(output_path + f"/nbest_{epoch}.json", 'w', encoding='utf-8') as f:
            json.dump(nbest_to_save, f, ensure_ascii=False)
        with open(output_path + f"/ner_pred_{epoch}.json", 'w', encoding='utf-8') as f:
            json.dump(ner_predictions, f, ensure_ascii=False)

        squad_acc_dict = self.processor.evaluate(predictions, nbest_predictions, test_examples)
        print("Macro NER: ", macro_ner)
        print("Micro NER: ", micro_ner)
        index_to_file = {q_id: example.file_name.split('.')[0] for example in test_examples for q_id in example.q_ids}
        index_to_question_text = {example.q_ids[i]: example.question_texts[i] for example in test_examples for i in
                                  range(len(example.q_ids))}
        all_filenames = sorted(list(set([index_to_file[q_id] for q_id in index_to_file])))
        result = {filename: {} for filename in all_filenames}
        for q_id in nbest_to_save:
            result[index_to_file[q_id]][index_to_question_text[q_id]] = [x['text'] for x in sorted(nbest_to_save[q_id],
                                                                                                   key=lambda x: x[
                                                                                                       'probability'],
                                                                                                   reverse=True)]
        if evaluation_function != None:
            nathan_score = evaluation_function(ca_data, result)
            squad_acc_dict.update(nathan_score)

        return squad_acc_dict, macro_ner, micro_ner, ner_detail, detailed_predictions

    def load(self, checkpoint_path):
        self.model.from_pretrained(checkpoint_path)

    def eval(self, data_dir, train_filename, test_filename, output_dir, train_bs, test_bs, num_train_epochs=10,
             cache_dir=None, override_cache=True, report_name=None, training_mode='NER', neptune_credential=None,
             evaluation_function=None, ca_data=None):

        documents = self.load_dataset(os.path.join(data_dir, test_filename))
        report_dict = {}
        for document in documents:
            print(f"Eval {document}")
            test_dataset = documents[document]['dataset']
            test_features = documents[document]['features']
            test_examples = documents[document]['examples']
            test_dataloader = DataLoader(test_dataset, batch_size=test_bs, num_workers=0, shuffle=False)
            squad_acc_dict, macro_ner, micro_ner, ner_detail, detailed_predictions = self.evaluate(test_dataloader,
                                                                                                   test_features,
                                                                                                   test_examples,
                                                                                                   0, training_mode,
                                                                                                   evaluation_function,
                                                                                                   ca_data)
            report_dict[document] = {'Accuracy': ner_detail,
                                     'prediction': detailed_predictions}

        with open("evaluate_detailed.json", 'w', encoding='utf-8') as f:
            json.dump(report_dict, f, ensure_ascii=False)


if __name__ == '__main__':
    from processors import prj_path

    # from neptune_credentials import fastqa_credential
    import os

    fold_prefix = 'fold_'
    train_file = f'train.json'
    test_file = 'test.json'
    data_dir = 'D:\\menu_extraction\\data/'
    label_dict_file = data_dir + 'question_list.json'  # can be list or dict

    output_dir = 'D:\\menu_extraction\\outputs'
    cache_dir = prj_path + f'/caches'

    all_labels = sorted([x for x in json.load(open(label_dict_file, 'r', encoding='utf-8'))])

    attention_type = 'Single'

    tokenizer_name = 'microsoft/layoutlm-base-uncased'
    pretrained_path = 'C:\\Users\\Levi\\Desktop\\model_epoch_15'

    trainer = Evaluator(pretrained_path, tokenizer_name, all_labels, embedding_from_encoder=False,
                        use_multiple_attention=attention_type == 'Multiple',
                        max_seq_len=128, doc_stride=32, version='v1')

    training_mode = 'QANER'
    batch_size = 8

    trainer.eval(data_dir=data_dir, train_filename=train_file, test_filename=test_file,
                 output_dir=output_dir,
                 train_bs=batch_size, test_bs=batch_size, num_train_epochs=10, cache_dir=cache_dir,
                 override_cache=True,
                 report_name=f'menu_v1_test_stable' + training_mode,
                 training_mode=training_mode,
                 # neptune_credential=fastqa_credential
                 )
