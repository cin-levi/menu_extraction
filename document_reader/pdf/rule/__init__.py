from .base_rules import BaseRule, NegativeBaseRule, PositiveBaseRule


class Rule:
    from .base_rules import NegativeBaseRule, PositiveBaseRule
    from .bullet_line import BulletLine
    from .numbering_line import NumberingLine
    from .dot_ending import DotEnding
    from .first_line import FirstLine
    from .font_different import FontDifferent
    from .indentation_difference import IndentationDifference
    from .line_distance import LineDistance
    from .line_length import LineLength
    from .opposite_alignment import OppositeAlignment
    from .indentation_shorter import IndentationShorter
    from .short_line import ShortLine
    from .title import Title
