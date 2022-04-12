from typing import List

from . import NegativeBaseRule
from ..line import Line
from ..paragraph import Paragraph
from ...utils import normalize_text


class Title(NegativeBaseRule):

    def run(self, paragraph: Paragraph, lines: List[Line], index: int,
            **kwargs):
        from ...utils.config import settings
        for patterns in settings.pdf.header.patterns:
            for pattern in patterns:
                text = normalize_text(lines[index].text)
                if pattern.search(text) is not None:
                    return self.RULE_CODE
