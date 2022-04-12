from typing import List

from . import NegativeBaseRule
from ..line import Line
from ..paragraph import Paragraph


class NumberingLine(NegativeBaseRule):
    """Line start with a numbering must be on a new paragraph"""

    def run(self, paragraph: Paragraph, lines: List[Line], index: int, **kwargs):
        if lines[index].numbering:
            return self.RULE_CODE
