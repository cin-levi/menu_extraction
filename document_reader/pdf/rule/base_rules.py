from typing import List

from ..line import Line
from ..paragraph import Paragraph


class BaseRule:
    def run(self, paragraph: Paragraph, lines: List[Line], index: int, **kwargs):
        """ Decide whether the line specified by `lines` and `index` belong to `paragraph`."""
        pass

    @property
    def name(self) -> str:
        return self.__class__.__name__


class NegativeBaseRule(BaseRule):
    @property
    def name(self) -> str:
        return super().name + '[-]'

    RULE_CODE = False

    def run(self, paragraph: Paragraph, lines: List[Line], index: int, **kwargs):
        """ Negative Rule states that the `lines[index]` shouldn't belong to `paragraph`.
        Return either False or None.
        """
        return self.RULE_CODE


class PositiveBaseRule(BaseRule):
    @property
    def name(self) -> str:
        return super().name + '[+]'

    RULE_CODE = True

    def run(self, paragraph: Paragraph, lines: List[Line], index: int, **kwargs):
        """ Positive Rule states that `lines[index]` should belong to `paragraph`
        Return either True or None
        """
        return self.RULE_CODE
