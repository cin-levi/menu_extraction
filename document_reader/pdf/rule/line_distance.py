import logging

from . import NegativeBaseRule

logger = logging.getLogger(__name__)


class LineDistance(NegativeBaseRule):
    """If the next line is far away from the current line, then it should start
    a new paragraph
    *************
    *************
    *************

    *************  <- line that far away from the previous one
    *************
    *****
    """

    def __init__(self, threshold: float = 1.2):
        """:param threshold: a factor of the height of the line"""
        self.threshold = threshold

    def run(self, paragraph, lines, index, **kwargs):
        line = lines[index]

        distance = float(line.rect['top'] - paragraph.rect['bottom'])
        line_height = line.rect['bottom'] - line.rect['top']
        threshold = self.threshold * float(line_height)
        if distance > threshold:
            return self.RULE_CODE

        if len(paragraph.lines) > 2:
            d = abs(paragraph.lines[-1].rect['top'] - \
                    paragraph.lines[-2].rect['bottom'])
            if distance > (float(d) * 1.25):
                return self.RULE_CODE
