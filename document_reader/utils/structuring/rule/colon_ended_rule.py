from . import BaseRule


class ColonEndedRule(BaseRule):
    def run(self, current_node, new_node) -> bool:
        if current_node.is_root():
            return False

        if current_node.paragraph.text.rstrip().endswith(':'):
            if current_node.can_has_child():
                current_node.add_child(new_node)
            else:
                current_node.parent.add_child(new_node)

            return True

        return False
