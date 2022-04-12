from . import NegativeBaseRule


class IndentationDifference(NegativeBaseRule):
    """Two consecutive lines with indentation different greater
    than a certain threshold should be on different paragraph"""

    def __init__(self, threshold=0.3):
        """:param threshold: in factor of the page width"""
        self.threshold = threshold

    def run(self, paragraph, lines, index, **kwargs):
        distance = abs(lines[index].rect['x0'] - paragraph.lines[-1].rect['x0'])
        threshold = float(kwargs['page'].width) * self.threshold

        if distance > threshold:
            return self.RULE_CODE
