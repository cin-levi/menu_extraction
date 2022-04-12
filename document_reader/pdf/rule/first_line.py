from . import PositiveBaseRule


class FirstLine(PositiveBaseRule):
    """The very first line of the document must start a new paragraph"""

    def run(self, paragraph, lines, index, **kwargs):
        if not paragraph:
            return self.RULE_CODE
