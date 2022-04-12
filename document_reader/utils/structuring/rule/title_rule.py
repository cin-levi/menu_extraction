from . import BaseRule


class TitleRule(BaseRule):
    """If the current node can have child, then the new node will be the child"""

    def run(self, current_node, new_node) -> bool:
        if current_node.can_has_child():
            current_node.add_child(new_node)

            return True

        return False
