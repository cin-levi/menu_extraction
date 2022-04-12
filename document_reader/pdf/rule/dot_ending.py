import logging
from typing import List

from . import NegativeBaseRule
from ..line import Line
from ..paragraph import Paragraph

logger = logging.getLogger(__name__)


class DotEnding(NegativeBaseRule):
    """If the previous line ended with a dot in `pattern`,
    then the current line could be on a different paragraph."""

    def __init__(self, pattern: str = '。.!?'):
        self.pattern = pattern

    def run(self, paragraph: Paragraph, lines: List[Line], index: int,
            **kwargs):
        if paragraph.text[-1] in self.pattern:
            from pdfplumber.utils import objects_to_rect
            page_x1 = objects_to_rect(kwargs['page'].chars)['x1']
            line_x1 = paragraph.lines[-1].rect['x1']
            if line_x1 != page_x1:
                return self.RULE_CODE
