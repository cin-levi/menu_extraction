from typing import List

from . import NegativeBaseRule
from ..line import Line
from ..paragraph import Paragraph


class BulletLine(NegativeBaseRule):
    """Line start with a bullet must be on a new paragraph"""

    def run(self, paragraph: Paragraph, lines: List[Line], index: int, **kwargs):
        if lines[index].bullet:
            return self.RULE_CODE
