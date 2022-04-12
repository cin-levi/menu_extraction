from . import BaseRule


class NaiveRule(BaseRule):
    """ Add the new node as a sibling of the current node.
    Note: consider using this rule as the last rule if you can not use IndentRule.
    """

    def run(self, current_node, new_node) -> bool:
        current_node.parent.add_child(new_node)
        return True
