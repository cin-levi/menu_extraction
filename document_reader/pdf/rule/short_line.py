from . import NegativeBaseRule


class ShortLine(NegativeBaseRule):
    """ If the previous line is too short then consider the next line is on a
    new paragraph
    **
    ************ <- previous line is too short
    ************
    ********
    """

    def __init__(self, threshold=0.5):
        """:param threshold: in factor of the page width"""
        self.threshold = threshold

    def run(self, paragraph, lines, index, **kwargs):
        width = paragraph.lines[-1].width
        threshold = float(kwargs['page'].width) * self.threshold

        if width < threshold:
            return self.RULE_CODE
