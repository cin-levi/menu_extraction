"""
The collection of rules for building a tree out of the input paragraphs of a document.
  * `BaseRule` is the base class for all the other rules.
  * Each rule has a `run` method that try to insert a new node into the tree
  * If a rule can find the parent for the new node, then it returns True.
  Otherwise, it returns False and the next rule will be run.
"""

from .base_rule import BaseRule


class StructuringRule:
    from .base_rule import BaseRule
    from .bullet_rule import BulletRule
    from .colon_ended_rule import ColonEndedRule
    from .first_paragraph_rule import FirstParagraphRule
    from .colon_ended_rule import ColonEndedRule
    from .toc_rule import TocRule
    from .title_rule import TitleRule
    from .table_rule import TableRule
    from .naive_rule import NaiveRule
    from .indent_rule import IndentRule
    from .numbering_rule import NumberingRule
    from .expanded_table_rule import ExpandedTableRule
