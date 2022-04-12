from transformers.models.layoutlm.modeling_layoutlm import *

_CONFIG_FOR_DOC = "LayoutLMConfig"
_TOKENIZER_FOR_DOC = "LayoutLMTokenizer"

import torch
from torch import nn
from torch.nn import CrossEntropyLoss


class NERLayoutLM(LayoutLMPreTrainedModel):

    def __init__(self, config: LayoutLMConfig, num_question, max_seq_len, init_embedding=None,
                 use_multiple_attention=True, embedding_from_encoder=True):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.layoutlm = LayoutLMModel(config)  # we can change it to bert here
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.max_seq_len = max_seq_len
        self.use_multiple_attention = use_multiple_attention
        self.num_question = num_question
        self.num_ner_label = num_question * 4 + 1  # B-, I-, O

        self.sequence_classifier = nn.Linear(config.hidden_size, config.num_labels)  # for text classification
        self.token_classifier = nn.Linear(config.hidden_size, self.num_ner_label)  # for sequence tagging

        init_embedding = torch.rand((self.num_question, config.hidden_size))
        self.question_embedding = nn.Embedding.from_pretrained(init_embedding)

        if self.use_multiple_attention:  # if  use multiple attention
            self.start_projections = nn.ModuleList(
                [nn.Linear(config.hidden_size, config.hidden_size, bias=False) for i in range(self.num_question)])
            self.end_projections = nn.ModuleList(
                [nn.Linear(config.hidden_size, config.hidden_size, bias=False) for i in range(self.num_question)])
        else:  # else use the same projection for all layer
            self.start_projections = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
            self.end_projections = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.init_weights()

    def set_question_embedding(self, embedding_matrix):
        self.question_embedding = nn.Embedding.from_pretrained(embedding_matrix)

    def get_input_embeddings(self):
        return self.layoutlm.embeddings.word_embeddings

    def get_question_projected(self, question_indexes, question_embedding, projections):
        bs = question_indexes.size()[0]
        batch_question_outputs = []
        # get question output
        for b in range(bs):
            q_batch = []
            for i, q_idx in enumerate(question_indexes[b]):
                q = q_idx.item()
                if self.use_multiple_attention:
                    q_batch.append(projections[q](question_embedding[b][i]))
                else:
                    q_batch.append(projections(question_embedding[b][i]))
            q_batch = torch.stack(q_batch)
            batch_question_outputs.append(q_batch)
        batch_question_outputs = torch.stack(batch_question_outputs)
        return batch_question_outputs

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            bbox=None,
            sequence_label_id=None,
            token_label_ids=None,
            question_indexes=None,
            start_positions=None,
            end_positions=None,
            # default parameter
            return_dict=None,
            **kwargs
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        parameters = {'input_ids': input_ids,
                      'bbox': bbox,
                      'attention_mask': attention_mask,
                      'token_type_ids': token_type_ids,
                      'return_dict': return_dict,
                      }
        # add layoutlm later
        outputs = self.layoutlm(**parameters)
        logits = outputs[0]
        logits = self.dropout(logits)  # bs x seq x h
        loss_fct = CrossEntropyLoss()
        # NER part
        ner_logits = self.token_classifier(logits)  # bs x seq x #labels
        # token_label_ids: bs x seq
        ner_loss = loss_fct(ner_logits.transpose(1, -1), token_label_ids)
        # FastQA part
        question_embedding = self.question_embedding(question_indexes)
        question_start_projected = self.get_question_projected(question_indexes, question_embedding,
                                                               self.start_projections)
        question_end_projected = self.get_question_projected(question_indexes, question_embedding, self.end_projections)

        logits_tranposed = logits.transpose(-1, -2)
        start_logits = torch.matmul(question_start_projected, logits_tranposed)
        end_logits = torch.matmul(question_end_projected, logits_tranposed)

        start_loss = loss_fct(start_logits.transpose(1, -1), start_positions)
        end_loss = loss_fct(end_logits.transpose(1, -1), end_positions)
        return (start_loss + end_loss, ner_loss,) + (start_logits, end_logits, ner_logits)
