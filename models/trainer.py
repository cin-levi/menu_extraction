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


class Trainer(object):
    def __init__(self, pretrained_path, tokenizer_name, all_labels, max_seq_len=384, doc_stride=128, max_ans_len=60,
                 embedding_from_encoder=False, use_multiple_attention=False, version='v1'):
        if version == 'v1':
            self.model = NERLayoutLM.from_pretrained(pretrained_path, num_question=len(all_labels),
                                                     max_seq_len=max_seq_len, init_embedding=None,
                                                     use_multiple_attention=use_multiple_attention,
                                                     embedding_from_encoder=embedding_from_encoder)
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=False)
        elif version == 'v2':
            self.model = NERLayoutLMv2.from_pretrained(pretrained_path, num_question=len(all_labels),
                                                       max_seq_len=max_seq_len, init_embedding=None,
                                                       use_multiple_attention=use_multiple_attention,
                                                       embedding_from_encoder=embedding_from_encoder)
            self.tokenizer = LayoutLMv2Tokenizer.from_pretrained(tokenizer_name)
        else:
            raise Exception(f"Do not support version {version}")

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

    def create_optimizer_and_scheduler(self, num_training_steps: int):
        """
        Setup the optimizer and the learning rate scheduler.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through :obj:`optimizers`, or subclass and override this method in a subclass.
        """
        if self.optimizer is None:
            no_decay = ["bias", "LayerNorm.weight"]
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                    "weight_decay": 0.0,
                },
                {
                    "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                    "weight_decay": 0.0,
                },
            ]
            self.optimizer = AdamW(
                optimizer_grouped_parameters,
                lr=5e-5,
                betas=(0.9, 0.999),
                eps=1e-8,
            )
        if self.lr_scheduler is None:
            self.lr_scheduler = get_linear_schedule_with_warmup(
                self.optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
            )

    def calculate_accuracy(self, logits, target):
        pass

    def load_dataset(self, data_path):
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)['data']
        examples = []
        print("get examples")
        for i, document in tqdm(enumerate(data)):
            title = document.get('title', "")
            for a_paragraph in document['paragraphs']:
                examples.append(self.processor.read_qa_examples(a_paragraph, title))
        print("convert example to features")
        features = self.processor.convert_examples_to_features(examples)
        dataset = self.processor.features_to_dataset(features)
        data_to_cache = {"examples": examples, "features": features, "dataset": dataset}
        return data_to_cache

    def train_epoch(self, train_dataloader, training_mode):
        # training_mode in ['FastQA', 'NER', 'FastQANER']
        print("training_mode: ", training_mode)
        self.model.train()
        epoch_loss = []
        for batch in tqdm(train_dataloader):
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
            qa_loss = outputs[0]
            ner_loss = outputs[1]
            if training_mode == 'QA':
                loss = qa_loss
            elif training_mode == 'NER':
                loss = ner_loss
            else:
                loss = qa_loss + ner_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)  # auto clip grad higher than 1
            self.optimizer.step()
            self.model.zero_grad()
            self.lr_scheduler.step()
            epoch_loss.append(loss.data)

        print(f" average_loss: {sum(epoch_loss) / len(epoch_loss)}")
        average_loss = sum(epoch_loss) / len(epoch_loss)
        return average_loss

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

        return squad_acc_dict, macro_ner, micro_ner, ner_detail

    def save(self, path_dir):
        self.model.save_pretrained(path_dir)
        with open(path_dir + '/metadata.json', 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, ensure_ascii=False)
        # construct metadata

    def load(self, checkpoint_path):
        self.model.from_pretrained(checkpoint_path)

    def train(self, data_dir, train_filename, test_filename, output_dir, train_bs, test_bs, num_train_epochs=10,
              cache_dir=None, override_cache=True, report_name=None, training_mode='NER', neptune_credential=None,
              evaluation_function=None, ca_data=None):
        if neptune_credential:
            import neptune.new as neptune
            neptune_run = neptune.init(**neptune_credential)
            neptune_run['experiment_name'] = report_name
            neptune_run['batch_size'] = train_bs
            neptune_run['lr'] = 5e-5
            neptune_run['training_mode'] = training_mode
        else:
            neptune_run = None
        assert training_mode in ['QA', 'NER', 'QANER']
        # TODO: get cached data
        cached_file_train = cache_dir + '/cached_train'
        cached_file_test = cache_dir + '/cached_test'
        if not override_cache and os.path.exists(cached_file_train):
            data_train_to_cache = torch.load(cached_file_train)
        else:
            data_train_to_cache = self.load_dataset(os.path.join(data_dir, train_filename))
            if cache_dir != None:
                print(f"cache training data to {cached_file_train}")
                torch.save(data_train_to_cache, cached_file_train)
        if not override_cache and os.path.exists(cached_file_test):
            data_test_to_cache = torch.load(cached_file_test)
        else:
            data_test_to_cache = self.load_dataset(os.path.join(data_dir, test_filename))
            if cache_dir != None:
                print(f"cache training data to {cached_file_test}")
                torch.save(data_test_to_cache, cached_file_test)
        train_dataset = data_train_to_cache['dataset']
        train_dataloader = DataLoader(train_dataset, batch_size=train_bs, num_workers=0,
                                      shuffle=True)  # in window, remove nw = 0. Change it in Linux

        test_dataset = data_test_to_cache['dataset']
        test_features = data_test_to_cache['features']
        test_examples = data_test_to_cache['examples']
        test_dataloader = DataLoader(test_dataset, batch_size=test_bs, num_workers=0, shuffle=False)

        num_update_steps_per_epoch = len(train_dataloader) // 1
        num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)

        max_steps = int(num_update_steps_per_epoch * num_train_epochs)

        self.optimizer = None
        self.lr_scheduler = None
        self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        report_dict = {}
        training_time = 0
        for i in range(1, num_train_epochs + 1):
            print(f"training epoch {i} ...")
            now = datetime.datetime.now()
            train_loss = self.train_epoch(train_dataloader, training_mode)
            epoch_time = datetime.datetime.now() - now
            if training_time == 0:
                training_time = epoch_time
            else:
                training_time = training_time + epoch_time
            report_dict[i] = {'epoch': i,
                              'epoch_time': str(epoch_time),
                              'training_time': str(training_time),
                              'train_loss': str(train_loss)}
            infer_now = datetime.datetime.now()
            squad_acc_dict, macro_ner, micro_ner, ner_detail = self.evaluate(test_dataloader, test_features,
                                                                             test_examples,
                                                                             i, training_mode,
                                                                             evaluation_function, ca_data)
            report_dict[i].update(squad_acc_dict)
            report_dict[i].update(macro_ner)
            report_dict[i].update(micro_ner)
            report_dict[i].update(ner_detail)
            print("evaluation: ")
            for x in report_dict[i]:
                if x not in ner_detail:
                    print(f"{x}: {report_dict[i][x]}")
            print("Detailed NER report: ")
            for x in ner_detail:
                print(f"{x}\tP: {ner_detail[x]['p']}\tR: {ner_detail[x]['r']}\tF: {ner_detail[x]['f']}")

            if i % SAVE_EVERY == 0:
                print("save check point ...")
                self.save(output_dir + f'/model_epoch_{i}')
        return report_dict

