import os, json, datetime
import numpy as np
import torch
from tqdm import tqdm
from datetime import datetime
from models.layoutlm_model import NERLayoutLM
from models.layoutlmv2_model import NERLayoutLMv2
from processors.fastqa_based_processor import CustomProcessor, Result, NerResult

from transformers import AutoTokenizer, BertConfig
from torch.utils.data import DataLoader

from processors import prj_path

cuda = torch.cuda.is_available()


class FastQAInference(object):
    def __init__(self, model=None, tokenizer=None, processor=None):
        self.model = model
        self.tokenizer = tokenizer
        self.processor = processor

        if cuda:
            self.model.cuda()



    def create_dataset(self, data):
        examples = []
        print("get examples")
        for i, document in tqdm(enumerate(data)):
            print(document)
            title = document.get('title', "")
            for a_paragraph in document['paragraphs']:
                examples.append(self.processor.read_qa_examples(a_paragraph, title))
        print("convert example to features")
        features = self.processor.convert_examples_to_features(examples)
        dataset = self.processor.features_to_dataset(features)
        data_to_cache = {"examples": examples, "features": features, "dataset": dataset}
        return data_to_cache

    def evaluate(self, test_dataloader, test_features, test_examples, calculate_acc, training_mode):
        # training_mode in ['FastQA', 'NER', 'FastQANER']
        if cuda:
            self.model.cuda()
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

        return ner_predictions

    def predict(self, json_data, test_bs, calculate_acc=False, training_mode='NER',
                num_workers=0):
        print("Start inference using input json file. Make sure that you file is in FastQA format!")
        print("json_data: ", json_data)
        data_test_to_cache = self.create_dataset(json_data)
        test_dataset = data_test_to_cache['dataset']
        test_features = data_test_to_cache['features']
        test_examples = data_test_to_cache['examples']
        test_dataloader = DataLoader(test_dataset, batch_size=test_bs, num_workers=num_workers, shuffle=False)
        report_dict = {}
        now = datetime.now()

        ner_predictions = self.evaluate(test_dataloader, test_features,
                                        test_examples,
                                        calculate_acc,
                                        training_mode)

        return ner_predictions

class LayoutLMInference(FastQAInference):
    @classmethod
    def load_from_checkpoint(cls, path):
        with open(path + '/metadata.json', 'r', encoding='utf-8') as f:
            metadata = json.load(f)

        tokenizer = AutoTokenizer.from_pretrained(metadata['tokenizer_name'], use_fast=False)

        processors = CustomProcessor(tokenizer=tokenizer,
                                     max_seq_len=metadata['max_seq_len'],
                                     max_query_len=60,
                                     max_ans_len=metadata['max_ans_len'],
                                     doc_stride=metadata['doc_stride'],
                                     question_list=metadata['question_list'],
                                     label_list=['0', '1'],  # can be use in classification mode
                                     version_2_with_negative=True)

        model = NERLayoutLM.from_pretrained(path, num_question=len(metadata['question_list']),
                                            max_seq_len=metadata['max_seq_len'],
                                            use_multiple_attention=metadata['use_multiple_attention'])
        return cls(model, tokenizer, processors)

class LayoutLMV2Inference(FastQAInference):
    @classmethod
    def load_from_checkpoint(cls, path):
        with open(path + '/metadata.json', 'r', encoding='utf-8') as f:
            metadata = json.load(f)

        from layoutlmv2.tokenization_layoutlmv2 import LayoutLMv2Tokenizer

        tokenizer = LayoutLMv2Tokenizer.from_pretrained(metadata['tokenizer_name'])

        processors = CustomProcessor(tokenizer=tokenizer,
                                     max_seq_len=metadata['max_seq_len'],
                                     max_query_len=60,
                                     max_ans_len=metadata['max_ans_len'],
                                     doc_stride=metadata['doc_stride'],
                                     question_list=metadata['question_list'],
                                     label_list=['0', '1'],  # can be use in classification mode
                                     version_2_with_negative=True)

        model = NERLayoutLMv2.from_pretrained(path, num_question=len(metadata['question_list']),
                                            max_seq_len=metadata['max_seq_len'],
                                            use_multiple_attention=metadata['use_multiple_attention'])
        return cls(model, tokenizer, processors)

if __name__ == '__main__':
    checkpoint_path = 'D:\\FastQA\\outputs\\best_model'
    output_path = prj_path + '/outputs'
    model = LayoutLMInference.load_from_checkpoint(checkpoint_path)
    test_file = 'D:\\FastQA\\data\\CinBidding\\AcceptAgumentation\\train.json'
    model.predict(test_file, 1, output_path)
