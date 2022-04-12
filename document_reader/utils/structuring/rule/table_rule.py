from . import BaseRule


class TableRule(BaseRule):
    """ Table row always take the direct above paragraph as parent
    """

    def run(self, current_node, new_node) -> bool:
        if new_node.paragraph.is_table():
            node = current_node
            if not node.can_has_child():
                node = node.parent

            node.add_child(new_node)

            return True

        return False
