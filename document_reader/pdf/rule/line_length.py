from . import NegativeBaseRule


class LineLength(NegativeBaseRule):
    """ If previous line is much shorter than current line, then it shouldn't belong to the paragraph.
            Example:
                aaaaaaaaaaaa
                aaaaaaaaaaaa
                aaaaaa
                aaaaaaaaaaaa <-- this one should start a new paragraph.
    """

    def __init__(self, threshold=0.3):
        """:param threshold: a factor of length of the line"""
        self.threshold = threshold

    def run(self, paragraph, lines, index, **kwargs):
        distance = lines[index].rect['x1'] - paragraph.lines[-1].rect['x1']
        if distance > self.threshold * float(kwargs['page'].width):
            return self.RULE_CODE
