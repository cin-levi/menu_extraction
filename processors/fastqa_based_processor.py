import argparse
import json
import os, re
import uuid
from dataclasses import dataclass
from enum import Enum
from functools import partial
from multiprocessing import Pool, cpu_count
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from sklearn.metrics import classification_report

import torch
from torch.utils.data import TensorDataset
from tqdm import tqdm
from transformers import PreTrainedTokenizer
from transformers.models.bert.tokenization_bert import whitespace_tokenize

from processors.base import Mode, Processor, InputExample, InputFeatures
from utils.qa_utils import (
    check_is_max_context,
    compute_predictions_log_probs,
    improve_answer_span,
    is_whitespace,
    squad_evaluate,
    fastqa_compute_predictions_logits,
    ner_compute_predictions_logits,
    ner_evaluate, ner_evaluate_by_char,
    combine_ner_and_QA_result
)


# from processors.question_embedded_utils import compute_predictions_logits, squad_evaluate


@dataclass
class Example(InputExample):
    doc_tokens: List[str]  # context was tokenized
    char_to_word_offset: List[int]  # convert from char to word
    q_ids: Optional[List[str]]  # list of question id, use for original FastQA
    question_texts: Optional[List[str]] = None  # list of question text, use for original FastQA
    positions: Optional[List[List[int]]] = None  # position of each words, use for layoutlm
    answers: Optional[List[Dict]] = None  # answers for each question, use for original FastQA
    start_positions: Optional[List[int]] = None  # use for original FastQA
    end_positions: Optional[List[int]] = None
    token_labels: Optional[List[str]] = None  # label of each words, use for NER
    sequence_label: str = None  # label of the whole sentence, use for classification
    file_name: str = None


@dataclass
class Features:
    paragraph_len: int
    paragraph_start: int
    example_uid: str
    # context part
    tokens: List[str]
    input_ids: List[int]
    positions: List[List[int]]  # if not use position, set it = [[0,0,0,0]]*len(tokens)
    token_is_max_context: List[bool]
    token_to_orig_map: Dict[int, int]  # now required
    # Question part
    q_ids: Optional[List[str]] = None  # id of question in example
    questions_indexes: Optional[List[int]] = None  # index of question in embedding matrix , use in original FastQA
    start_positions: Optional[List[int]] = None
    end_positions: Optional[
        List[int]] = None  # if FastQA, len(q_ids) = len(questions_indexes) = len(start_positions) = len(end_positions)
    # Other task part
    token_label_ids: Optional[List[int]] = None  # For NER
    sequence_label_id: Optional[int] = None  # For classification
    # information from tokenizer
    attention_mask: Optional[List[int]] = None
    token_type_ids: Optional[List[int]] = None
    # Unchange parameter, but you can change it if necessary
    cls_index: Optional[int] = None
    p_mask: Optional[List[int]] = None
    tok_start_to_orig_index: Optional[Dict[int, int]] = None
    tok_end_to_orig_index: Optional[Dict[int, int]] = None


@dataclass
class Result:
    uid: str
    start_logits: Optional[torch.Tensor] = None  # for FastQA
    end_logits: Optional[torch.Tensor] = None  # for FastQA
    start_top_index: Optional[torch.Tensor] = None
    end_top_index: Optional[torch.Tensor] = None


@dataclass
class NerResult:
    uid: str
    logits: Optional[torch.Tensor] = None  # for NER
    cls_logits: Optional[torch.Tensor] = None  # for Cls
    gt: Optional[torch.Tensor] = None


class CustomProcessor(Processor):
    def __init__(
            self,
            tokenizer: Union[PreTrainedTokenizer, str],
            max_seq_len: int,
            max_query_len: int,
            max_ans_len: int,
            doc_stride: int,
            question_list: List[str] = None,
            nbest_size: int = 20,
            null_score_diff_threshold: float = 0.0,
            version_2_with_negative: bool = False,
            **kwargs,
    ):

        kwargs["label_list"] = ["start", "end"]
        super().__init__(
            tokenizer=tokenizer,
            max_seq_len=max_seq_len,
            **kwargs,
        )
        self.max_query_len = max_query_len
        self.max_ans_len = max_ans_len
        self.doc_stride = doc_stride
        self.question_list = question_list  # sorted list
        # construct ner label dict
        self.ner_label_dict = {}
        self.ner_label_dict['O'] = 0
        prefix_offset = {1: 'B-', 2: 'I-', 3: 'E-', 4: 'S-'}
        for i in range(len(self.question_list)):
            for j in prefix_offset:
                self.ner_label_dict[prefix_offset[j] + self.question_list[i]] = 4 * i + j
        self.index_to_ner_label = {self.ner_label_dict[x]: x for x in self.ner_label_dict}
        self.question_dict = {question_list[i]: i for i in range(len(self.question_list))}
        self.nbest_size = nbest_size
        self.null_score_diff_threshold = null_score_diff_threshold
        self.version_2_with_negative = version_2_with_negative

    def read_qa_examples(self, paragraph, title=None) -> Example:
        if 'uid' in paragraph:
            paragraph_uid = paragraph['uid']
        else:
            paragraph_uid = str(uuid.uuid4())
        context_text = paragraph["context"]
        doc_tokens = []
        char_to_word_offset = []
        prev_is_whitespace = True
        for c in context_text:
            if is_whitespace(c):
                prev_is_whitespace = True
            else:
                if prev_is_whitespace:
                    doc_tokens.append(c)
                else:
                    doc_tokens[-1] += c
                prev_is_whitespace = False
            char_to_word_offset.append(len(doc_tokens) - 1)

        if 'position' in paragraph:
            assert len(paragraph['position']) == len(doc_tokens), f"{len(paragraph['position'])}  {len(doc_tokens)} {doc_tokens}"
        else:
            paragraph['position'] = [[0, 0, 0, 0]] * len(doc_tokens)

        token_labels = ['O'] * len(doc_tokens)
        # TODO: in this strategy, we will set priority in the index of question. So the label of entity will be assigned once only

        question_texts = []
        answers = []
        q_ids = []
        start_positions = []
        end_positions = []

        for qa in paragraph["qas"]:
            uid = qa["id"]
            question_text = qa["question"]

            is_impossible = qa.get("is_impossible", False)

            if not is_impossible:
                answer = qa["answers"][0]
                answer_text = answer["text"]
                assert len(answer_text) > 0, f"{uid}"

                assert context_text[answer["answer_start"]:].strip().startswith(answer_text), \
                    f"{context_text[answer['answer_start']:answer['answer_start'] + len(answer_text)]}, {answer_text}"
                start_position_character = answer["answer_start"]
                start_position = char_to_word_offset[start_position_character]
                end_position = char_to_word_offset[
                    min(
                        start_position_character + len(answer_text) - 1,
                        len(char_to_word_offset) - 1,
                    )
                ]

                actual_text = " ".join(doc_tokens[start_position: (end_position + 1)])
                cleaned_answer_text = " ".join(whitespace_tokenize(answer_text))

                if actual_text.find(cleaned_answer_text) == -1:
                    print(
                        f"Could not find answer: '{actual_text}' vs. '{cleaned_answer_text}' in example: {uid} with start is {start_position} and end is {end_position}"
                    )
                    print("char_to_word_offset: ", char_to_word_offset)
                    print("start_position_character: ", start_position_character)
                    1 / 0
            else:
                start_position = -1
                end_position = -1
            question_texts.append(question_text)
            answers.append(qa["answers"])
            q_ids.append(uid)
            start_positions.append(start_position)
            end_positions.append(end_position)

            # add label for each tokens
            for ner_answer in qa["answers"]:
                answer_text = ner_answer["text"]

                assert context_text[ner_answer["answer_start"]:].strip().startswith(
                    answer_text.strip()), f"{context_text[ner_answer['answer_start']:ner_answer['answer_start'] + len(answer_text)]}|{answer_text}"
                ner_start_position_character = ner_answer["answer_start"]

                # print("start_position_character: ", start_position_character)

                ner_start_position = char_to_word_offset[ner_start_position_character]
                ner_end_position = char_to_word_offset[
                    min(
                        ner_start_position_character + len(answer_text) - 1,
                        len(char_to_word_offset) - 1,
                    )]
                if not any([token_labels[i] != 'O' for i in range(ner_start_position, ner_end_position + 1)]):
                    # if there is no duplicated entity yet, then assign label for it
                    if ner_start_position == ner_end_position:
                        token_labels[ner_start_position] = 'S-' + question_text
                    else:
                        token_labels[ner_start_position] = 'B-' + question_text
                        token_labels[ner_start_position + 1:ner_end_position + 1] = ['I-' + question_text] * (
                                ner_end_position - ner_start_position - 1) + ['E-' + question_text]

        example = Example(
            uid=paragraph_uid,
            doc_tokens=doc_tokens,
            char_to_word_offset=char_to_word_offset,
            q_ids=q_ids,
            question_texts=question_texts,
            positions=paragraph['position'],
            answers=answers,
            start_positions=start_positions,
            end_positions=end_positions,
            token_labels=token_labels,
            file_name=title
        )
        return example

    def convert_examples_to_features(
            self, examples: List[Example], is_training: bool = True
    ) -> List[Features]:
        # with Pool(cpu_count()) as p:
        with Pool(1) as p:  # I tested the model in window therefore can not use pool
            featurize_func = partial(self._convert_example_to_features)
            feature_batches = list(
                tqdm(
                    p.imap(featurize_func, examples, chunksize=32),
                    total=len(examples),
                    desc="Convert SQuAD examples to features",
                )
            )
        features = []
        unique_id = 0
        for example_features in tqdm(feature_batches, total=len(feature_batches), desc="Add features' unique id"):
            if len(example_features) == 0:
                continue
            for feature in example_features:
                feature.uid = str(unique_id)
                features.append(feature)
                unique_id += 1
        return features

    def features_to_dataset(self, features: List[Features], mode: Union[str, Mode] = Mode.train) -> TensorDataset:
        dataset = [
            torch.tensor([f.input_ids for f in features], dtype=torch.long),
            torch.tensor([f.attention_mask for f in features], dtype=torch.long),
            torch.tensor([f.token_type_ids for f in features], dtype=torch.long),
            torch.tensor([f.positions for f in features], dtype=torch.long),
            torch.tensor([f.sequence_label_id for f in features], dtype=torch.long),
            torch.tensor([f.token_label_ids for f in features], dtype=torch.long),
            torch.tensor([f.questions_indexes for f in features], dtype=torch.long),
            torch.tensor([f.start_positions for f in features], dtype=torch.long),
            torch.tensor([f.end_positions for f in features], dtype=torch.long),
        ]
        dataset.append(torch.arange(dataset[0].size(0), dtype=torch.long))  # feature_indices
        return TensorDataset(*dataset)

    def _convert_example_to_features(self, example: Example):  # noqa: C901
        features = []
        tok_to_orig_index = []
        orig_to_tok_index = []
        all_doc_tokens = []
        all_positions = []
        all_doc_labels = []

        for (i, token) in enumerate(example.doc_tokens):
            orig_to_tok_index.append(len(all_doc_tokens))
            sub_tokens = self.tokenizer.tokenize(token)
            # assert len(sub_tokens) == 1, f"{token}, {sub_tokens}"
            # TODO: again, check sub tokens

            all_positions.extend([example.positions[i]] * len(sub_tokens))  # only one sub token for each token
            for sub_token in sub_tokens:
                tok_to_orig_index.append(i)
                all_doc_tokens.append(sub_token)
            token_label = example.token_labels[i].split('-', 1)[-1]
            prior = example.token_labels[i].split('-', 1)[0]
            if prior == 'B':
                sub_labels = [example.token_labels[i]] + ['I-' + token_label] * (len(sub_tokens) - 1)
            elif prior == 'S' and len(sub_tokens) > 1:
                sub_labels = ['B-' + token_label] + ['I-' + token_label] * (len(sub_tokens) - 2) + ['E-' + token_label]
            elif prior == 'E':
                sub_labels = ['I-' + token_label] * (len(sub_tokens) - 1) + ['E-' + token_label]
            else:
                sub_labels = [example.token_labels[i]] * len(sub_tokens)
            all_doc_labels.extend(sub_labels)

        question_indexes = []
        start_positions = []
        end_positions = []
        answer_texts = []
        is_impossibles = []
        q_ids = []
        for i in range(len(example.question_texts)):
            if example.question_texts[i] not in self.question_dict:
                print("Got unexpected question: ", example.question_texts[i])
            else:
                question_indexes.append(self.question_dict[example.question_texts[i]])
                if len(example.answers[i]) == 0:
                    is_impossibles.append(True)
                else:
                    is_impossibles.append(False)

                tok_start_position = orig_to_tok_index[example.start_positions[i]]
                assert example.start_positions[i] <= len(example.doc_tokens) - 1
                if example.end_positions[i] < len(example.doc_tokens) - 1:
                    tok_end_position = orig_to_tok_index[example.end_positions[i] + 1] - 1
                else:
                    tok_end_position = len(all_doc_tokens) - 1
                if len(example.answers[i]) > 0:
                    (tok_start_position, tok_end_position) = improve_answer_span(
                        all_doc_tokens,
                        tok_start_position,
                        tok_end_position,
                        self.tokenizer,
                        example.answers[i][0]['text'],
                    )

                start_positions.append(tok_start_position)
                end_positions.append(tok_end_position)
                answer_text = example.answers[i][0]['text'] if len(example.answers[i]) > 0 else ''
                if not is_impossibles[-1]:
                    assert len(answer_text) > 0
                answer_texts.append(answer_text)
                q_ids.append(example.q_ids[i])
                # check here
                answer_tokens = all_doc_tokens[tok_start_position: tok_end_position + 1]
                text = self.tokenizer.convert_tokens_to_string(answer_tokens)
                # if '[UNK]' not in text:
                #     if text != 'phm ngc linh':
                #
                #         assert re.sub('\s+', '', text) == re.sub('\s+', '',
                #                                                  answer_texts[-1]), \
                #             f"{text} || answer_text: {example.answers[i][0]['text']} || \n {answer_text} || \n start:  {example.start_positions[i]}" \
                #             f" || \n end: {example.end_positions[i]} || \n tok_start:  {tok_start_position} || \n tok_end: {tok_end_position}" \
                #             f" || \n len_doc_tokens: {len(example.doc_tokens)}"

        # todo: add missing question
        for question in self.question_dict:
            assert question in example.question_texts, f"{question}, {example.question_texts}"

        assert len(all_positions) == len(all_doc_tokens)
        assert len(question_indexes) == len(start_positions)
        assert len(question_indexes) == len(end_positions)

        num_added_tokens = self.tokenizer.num_special_tokens_to_add()
        stride = self.max_seq_len - self.doc_stride - num_added_tokens

        all_tokens = all_doc_tokens
        span_tokens = all_doc_tokens
        spans = []

        while span_tokens:
            encoded_dict = self.tokenizer.encode_plus(
                span_tokens,
                padding="max_length",
                truncation="only_first",
                max_length=self.max_seq_len,
                return_token_type_ids=True,
                return_overflowing_tokens=True,
                stride=stride,
            )

            if self.tokenizer.pad_token_id in encoded_dict["input_ids"]:
                non_padded_ids = encoded_dict["input_ids"][
                                 : encoded_dict["input_ids"].index(self.tokenizer.pad_token_id)
                                 ]
            else:
                non_padded_ids = encoded_dict["input_ids"]

            tokens = self.tokenizer.convert_ids_to_tokens(non_padded_ids)

            # TODO: add to get the answer
            paragraph_len = min(
                len(all_doc_tokens) - len(spans) * self.doc_stride,
                self.max_seq_len - num_added_tokens,
            )
            token_to_orig_map = {}
            curr_tok_start_to_orig_index = {}
            curr_tok_end_to_orig_index = {}
            for i in range(paragraph_len):
                index = 1 + i  # 1 for [CLS]
                token_to_orig_map[index] = tok_to_orig_index[len(spans) * self.doc_stride + i]  # for bert only

            encoded_dict["start"] = len(spans) * self.doc_stride
            encoded_dict["length"] = min(
                self.max_seq_len - num_added_tokens, len(all_tokens) - encoded_dict["start"]
            )
            padding_label_len = self.max_seq_len - num_added_tokens - encoded_dict[
                "length"]  # if smaller than the len of sequence, then pad with O
            token_labels = ['O'] + all_doc_labels[encoded_dict["start"]:encoded_dict["start"] + encoded_dict["length"]] \
                           + ['O'] * (padding_label_len + 1)

            encoded_dict["token_labels"] = [self.ner_label_dict[l] for l in token_labels]

            encoded_dict["token_to_orig_map"] = token_to_orig_map
            encoded_dict["tok_start_to_orig_index"] = curr_tok_start_to_orig_index
            encoded_dict["tok_end_to_orig_index"] = curr_tok_end_to_orig_index
            encoded_dict["token_is_max_context"] = {}

            encoded_dict["tokens"] = tokens
            encoded_dict["truncated_query_with_special_tokens_length"] = 1  # For [CLS] only

            end = encoded_dict["start"] + encoded_dict["length"]

            spans.append(encoded_dict)
            span_tokens = encoded_dict.get("overflowing_tokens", None)
        for span_id, span in enumerate(spans):
            # span["token_is_max_context"] = [False] * len(span["input_ids"])
            for token_id in range(span["length"]):
                span["token_is_max_context"][token_id + 1] = check_is_max_context(
                    spans, span_id, span["start"] + token_id
                )
            span_start = span['start']
            span_end = span['start'] + span['length'] - 1

            position_pad = [0, 0, 0, 0]
            positions = [position_pad] + all_positions[span_start:span_end + 1]
            if len(positions) < self.max_seq_len:
                positions += [position_pad] * (self.max_seq_len - len(positions) - 1)
            positions += [position_pad]
            assert len(positions) == len(span["input_ids"]), f"{len(positions)}, {len(span['input_ids'])}"

            # TODO: add q-a here
            q_idxes = []
            starts = []
            ends = []
            # assert len(question_indexes) == len(self.question_dict)
            for i in range(len(question_indexes)):
                curr_start = start_positions[i]
                curr_end = end_positions[i]

                out_of_span = False
                is_impossible = is_impossibles[i]

                if not (curr_start >= span_start and curr_end <= span_end):
                    out_of_span = True

                if out_of_span or is_impossible:
                    start_position = 0
                    end_position = 0
                    span_is_impossible = True
                else:

                    start_position = curr_start - span_start + 1
                    end_position = curr_end - span_start + 1

                if not (out_of_span or is_impossible):
                    answer_tokens = span['input_ids'][start_position: end_position + 1]
                    text = self.tokenizer.convert_tokens_to_string(self.tokenizer.convert_ids_to_tokens(answer_tokens))
                    # if '[UNK]' not in text and '<unk>' not in text:
                    #     if text != 'phm ngc linh':
                    #         assert text.replace(' ', '') == answer_texts[
                    #         i].strip().replace(' ', ''), f"{text} - {answer_texts[i]} |  len: {len(answer_texts[i])}"

                    # print(f"{text} - {answer_texts[i].strip()}")

                q_idxes.append(question_indexes[i])
                starts.append(start_position)
                ends.append(end_position)

            # TODO: remove duplicate, make sure there is only one instance of each question
            # We can remove this one but actually it do not effect the processor. I am not sure this part can help, but I keep it for now
            all_unique_q_idexes = list(set(q_idxes))
            assert len(all_unique_q_idexes) == len(self.question_dict)

            question_indexes_unique = []

            start_positions_unique = []
            end_positions_unique = []
            q_ids_unique = []

            remove_index = []
            for i in range(len(q_idxes)):
                if i in remove_index:
                    continue
                q_id = q_idxes[i]

                dupplicated_positions = [j for j in range(len(q_idxes)) if q_idxes[j] == q_id]
                if len(dupplicated_positions) == 1:
                    question_indexes_unique.append(q_idxes[i])
                    start_positions_unique.append(starts[i])
                    end_positions_unique.append(ends[i])
                    q_ids_unique.append(q_ids[i])
                else:
                    duplicated_ends = dupplicated_positions
                    non_zero_duplicated_ends = [j for j in duplicated_ends if ends[j] != 0]
                    if len(non_zero_duplicated_ends) > 0:
                        accpeted_idx = non_zero_duplicated_ends[0]
                    else:
                        accpeted_idx = duplicated_ends[0]
                    remove_index.extend(duplicated_ends)
                    question_indexes_unique.append(q_idxes[accpeted_idx])
                    start_positions_unique.append(starts[accpeted_idx])
                    end_positions_unique.append(ends[accpeted_idx])
                    q_ids_unique.append(q_ids[accpeted_idx])

            assert len(question_indexes_unique) == len(
                self.question_dict), f"{len(question_indexes_unique)}, {len(self.question_list)}, {question_indexes_unique}"

            features.append(
                Features(
                    paragraph_len=span["length"],
                    paragraph_start=span['start'],
                    example_uid=example.uid,
                    tokens=span["tokens"],
                    input_ids=span["input_ids"],
                    positions=positions,
                    token_is_max_context=span["token_is_max_context"],
                    token_to_orig_map=span['token_to_orig_map'],
                    q_ids=q_ids_unique,
                    questions_indexes=question_indexes_unique,
                    start_positions=start_positions_unique,
                    end_positions=end_positions_unique,
                    token_label_ids=span['token_labels'],
                    sequence_label_id=0,
                    attention_mask=span["attention_mask"],
                    token_type_ids=span["token_type_ids"],
                    cls_index=None,
                    p_mask=None,
                    tok_start_to_orig_index=span['tok_start_to_orig_index'],
                    tok_end_to_orig_index=span['tok_end_to_orig_index']
                )
            )
        return features

    def parse_inference_inputs(self, inputs: List[Dict[str, Any]]) -> List[Example]:
        examples = []
        for paragraph in inputs:
            examples.extend(self._parse_paragraph_to_examples(paragraph, title="inference", mode=Mode.inference))

        return examples

    def evaluate(self, predictions, nbest_predictions, examples) -> Dict[str, Any]:
        return squad_evaluate(examples, predictions, 'FastQA')

    def compute_predictions(
            self,
            examples: List[Example],
            features: List[Features],
            results: List[Result],
            version_2_with_negative=None,
            start_n_top=None,
            end_n_top=None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Compute predictions and nbest predictions from model's output

        Args:
            examples: List of :class:`~nlp.processors.SquadExample`.
            features: List of :class:`~nlp.processors.SquadFeatures`.
            results: List of :class:`~nlp.processors.SquadResult`.
            version_2_with_negative: if ``True``, the predictions will contain null predictions if possible.
                Default to ``None``. By default, it will use the processor ``version_2_with_negative`` param.
            start_n_top: ``start_n_top`` parameter for XLNet or XLM. Defaults to ``None``.
            end_n_top: ``end_n_top`` paramter for XLNet or XLM. Defaults to ``None``.

        Returns:
            predictions and nbest predictions
        """
        if start_n_top is not None and end_n_top is not None:
            # Compute predictions in XLNet, XLM's way
            return compute_predictions_log_probs(
                examples, features, results, self.tokenizer, self.max_ans_len, self.nbest_size, start_n_top, end_n_top
            )
        else:
            return fastqa_compute_predictions_logits(
                examples,
                features,
                results,
                self.tokenizer,
                self.max_ans_len,
                self.nbest_size,
                self.null_score_diff_threshold,
                (self.version_2_with_negative if version_2_with_negative is None else version_2_with_negative),
            )

    def ner_evaluate_by_field(self, predictions, examples) -> Dict[str, Any]:
        return ner_evaluate(predictions, examples, self.question_list)

    def ner_evaluate_by_char(self, predictions, examples) -> Dict[str, Any]:
        return ner_evaluate_by_char(predictions, examples, self.question_list)

    def compute_ner_predictions(
            self,
            examples: List[Example],
            features: List[Features],
            results: List[NerResult]
    ):
        """Compute predictions and nbest predictions from model's output

        Args:
            examples: List of :class:`~nlp.processors.SquadExample`.
            features: List of :class:`~nlp.processors.SquadFeatures`.
            results: List of :class:`~nlp.processors.SquadResult`.
            version_2_with_negative: if ``True``, the predictions will contain null predictions if possible.
                Default to ``None``. By default, it will use the processor ``version_2_with_negative`` param.
            start_n_top: ``start_n_top`` parameter for XLNet or XLM. Defaults to ``None``.
            end_n_top: ``end_n_top`` paramter for XLNet or XLM. Defaults to ``None``.

        Returns:
            predictions and nbest predictions
        """
        ner_predictions_logits = ner_compute_predictions_logits(
            features,
            results,
            self.tokenizer
        )
        # decode ner here
        id_to_example = {example.uid: example for example in examples}
        ner_outpu_json = {}
        for id in ner_predictions_logits:
            example = id_to_example[id]
            result = ner_predictions_logits[id]
            logits = result['logits']
            token_to_original_map = result['token_to_original_map']
            max_index = max([token_to_original_map[i] for i in token_to_original_map])

            assert max_index == len(example.doc_tokens) - 1
            predicted_labels = np.argmax(logits, axis=1)
            predicted_labels = [self.index_to_ner_label[predicted_labels[i]] for i in range(len(predicted_labels))]
            ner_outpu_json[id] = self.decode_ner(predicted_labels, token_to_original_map, example)
        return ner_outpu_json

    def decode_ner(self, predicted_labels, token_to_original_map, example):
        entities = []
        start = -1
        curent_label = 'O'
        for i in range(len(predicted_labels)):
            token_label = predicted_labels[i]
            token_label = token_label.split('-')[-1]
            prior = predicted_labels[i].split('-')[0]
            if prior == 'S':
                start = i
                curent_label = token_label
                answer_start = token_to_original_map[start]
                text = self.tokenizer.convert_tokens_to_string(example.doc_tokens[answer_start: answer_start + 1])
                position = example.positions[answer_start: answer_start+1]
                entities.append({'entity_type': curent_label, 'start': answer_start, 'text': text, 'position': position})
                start = -1
                curent_label = 'O'
            elif (prior == 'E') and (token_label == curent_label):
                if start != -1:
                    answer_start = token_to_original_map[start]
                    answer_end = token_to_original_map[i]
                    text = self.tokenizer.convert_tokens_to_string(example.doc_tokens[answer_start: answer_end + 1])
                    position = example.positions[answer_start: answer_end + 1]
                    entities.append({'entity_type': curent_label, 'start': answer_start, 'text': text, 'position': position})
                    start = -1
                    curent_label = 'O'
            elif (prior == 'I') and (token_label == curent_label):
                continue
            elif prior == 'B':
                start = i
                curent_label = token_label
            else:
                start = -1
                curent_label = 'O'
        return entities

    def get_entities(self, tokens, token_labels):

        entities = []
        current_text = ''
        curent_label = 'O'
        for i in range(len(tokens)):
            # token_label = self.index_to_ner_label[token_labels[i]]
            token_label = token_labels[i]
            token_label = token_label.split('-')[-1]
            if (token_label == curent_label) and (curent_label != 'O'):
                current_text += ' ' + tokens[i]
            else:
                if len(current_text) > 0:
                    entities.append({'entity_type': curent_label, 'entity_text': current_text})
                if token_label != 'O':
                    current_text = tokens[i]
                else:
                    current_text = ''
                curent_label = token_label
        if len(current_text) > 0:
            entities.append({'entity_type': curent_label, 'entity_text': current_text})
        return entities

    def evaluate_ner_without_decoding(self, ner_results):
        gts = []
        preds = []
        for result in ner_results:
            gt = result.gt
            logits = result.logits
            pred = np.argmax(logits, axis=1)
            gts.extend(gt)
            preds.extend(pred)
        print(classification_report(gts, preds))
        report = classification_report(gts, preds, output_dict=True)
        return report

    def combine_ner_and_QA_result(self, ner_results, qa_results, test_examples):
        return combine_ner_and_QA_result(ner_results, qa_results, test_examples)

    def generate_question_input_dict(self):
        question_inputs = self.tokenizer(self.question_list, padding=True)
        return question_inputs


if __name__ == '__main__':
    from processors import prj_path

    data_file = prj_path + '/data/train.json'
    from transformers import AutoTokenizer

    question_list_file = prj_path + '/data/question_list.json'

    all_labels = json.load(open(question_list_file, 'r', encoding='utf-8'))

    tokenizer = AutoTokenizer.from_pretrained('microsoft/layoutlm-base-uncased', use_fast=False)
    print("tokenize: ", tokenizer)
    processor = CustomProcessor(tokenizer=tokenizer,
                                max_seq_len=384,
                                max_query_len=60,
                                max_ans_len=60,
                                question_list=all_labels,
                                doc_stride=128,
                                data_dir='D:\\layoutlm\\data_qa',
                                label_list=['0', '1'],
                                train_filename='train.json')

    with open(data_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    features = []
    for doc in tqdm(data['data']):
        for a_paragraph in doc['paragraphs']:
            example = processor.read_qa_examples(a_paragraph)
            entities = processor.get_entities(example.doc_tokens, example.token_labels)
            features.extend(processor._convert_example_to_features(example))
    print(features[0])

    dataset = processor.features_to_dataset(features)
