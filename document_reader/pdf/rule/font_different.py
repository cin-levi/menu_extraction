import logging
from typing import List

from . import NegativeBaseRule
from ..line import Line
from ..paragraph import Paragraph

logger = logging.getLogger(__name__)


class FontDifferent(NegativeBaseRule):
    """Consecutive lines with different fonts would be in separated paragraph"""

    def __init__(self, size_threshold=2, ignore_font_face=True):
        self.ignore_font_name = ignore_font_face
        self.size_threshold = size_threshold

    def run(self, paragraph: Paragraph, lines: List[Line], index: int, **kwargs):
        size_diff = abs(paragraph.last_line.height - lines[index].height)
        has_different_font_face = paragraph.get_last_char_font().face != lines[index].get_last_char_font().face
        if size_diff > self.size_threshold \
                or (not self.ignore_font_name and has_different_font_face):
            return self.RULE_CODE
