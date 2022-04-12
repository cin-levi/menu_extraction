import abc
import argparse
import inspect
from argparse import ArgumentParser, Namespace
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from torch.utils.data import Dataset
from transformers import AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast


@dataclass
class InputExample:
    """A single training/test example.

    Args:
        uid: Unique id for the example.
    """
    uid: str


@dataclass
class InputFeatures:
    """A single set of features of data.
    Property names are the same names as the corresponding inputs to a model.

    Args:
        input_ids: Indices of input sequence tokens in the vocabulary.
        attention_mask: Mask to avoid performing attention on padding token indices.
            Mask values selected in ``[0, 1]``:
            Usually  ``1`` for tokens that are NOT MASKED, ``0`` for MASKED (padded) tokens.
        token_type_ids: (Optional) Segment token indices to indicate first and second
            portions of the inputs. Only some models use them.
    """

    attention_mask: Optional[List[int]] = None
    token_type_ids: Optional[List[int]] = None


class Mode(Enum):
    train = "train"
    dev = "dev"
    test = "test"
    inference = "inference"


class Processor(abc.ABC):

    def __init__(
        self,
        tokenizer: Union[PreTrainedTokenizer, str],
        max_seq_len: int,
        label_list: List[str],
        **kwargs
    ):
        if isinstance(tokenizer, PreTrainedTokenizer):
            self.tokenizer = tokenizer
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer, **kwargs)
        self.max_seq_len = max_seq_len

        if isinstance(label_list, str):
            label_list = label_list.split(",")
        self.label_list = label_list



    @abc.abstractmethod
    def convert_examples_to_features(self, examples):
        """Generate input features from examples"""
        raise NotImplementedError()

    @abc.abstractmethod
    def features_to_dataset(self, features: List[InputFeatures], mode: Union[str, Mode]) -> Dataset:
        """Get Pytorch Dataset object from list of input features"""
        raise NotImplementedError()

    @abc.abstractmethod
    def evaluate(self, *args, **kwargs) -> Dict[str, Any]:
        """Evaluate metrics for the dataset"""
        raise NotImplementedError()
