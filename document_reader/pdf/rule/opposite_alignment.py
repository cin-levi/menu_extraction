from typing import List

from . import NegativeBaseRule
from ..line import Line
from ..paragraph import Paragraph


class OppositeAlignment(NegativeBaseRule):
    """Two consecutive lines with opposite alignment, left vs right alignment, must be on different paragraph"""

    def run(self, paragraph: Paragraph, lines: List[Line], index: int, **kwargs):
        from document_reader.pdf.style import ALIGNMENT
        if {lines[index].alignment, paragraph.alignment} == {ALIGNMENT.LEFT, ALIGNMENT.RIGHT}:
            return self.RULE_CODE
