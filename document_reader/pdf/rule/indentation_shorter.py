import logging
from typing import List

from . import NegativeBaseRule
from ..line import Line
from ..paragraph import Paragraph

logger = logging.getLogger(__name__)


class IndentationShorter(NegativeBaseRule):
    """ If the line has bigger indentation than the previous one, they must not
    be in the same paragraph:
    ***********
    ***********
      ********* <- bigger indentation
    ***********
    """

    def __init__(self, threshold=0.02):
        """:param threshold: in factor of the page width"""
        self.threshold = threshold

    def run(self, paragraph: Paragraph, lines: List[Line], index: int,
            **kwargs):
        distance = paragraph.lines[-1].rect['x0'] - lines[index].rect['x0']
        threshold = float(kwargs['page'].width) * self.threshold

        if distance > threshold:
            return self.RULE_CODE
