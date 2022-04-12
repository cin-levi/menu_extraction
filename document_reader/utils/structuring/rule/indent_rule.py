from . import BaseRule


class IndentRule(BaseRule):
    """ Find the parent base on indentation of the nodes.
    Note: This rule always returns true, so any rule after this one on the list will not be run.
    """

    INDENTATION_DIFF_THRESHOLD = 0.5

    def run(self, current_node, new_node) -> bool:
        node = current_node

        if node.paragraph.is_header() or (
                not new_node.can_has_child() and node.can_has_child()):
            node.add_child(new_node)
            return True

        if abs(float(node.paragraph.indentation) - float(
                new_node.paragraph.indentation)) < self.INDENTATION_DIFF_THRESHOLD:
            node.parent.add_child(new_node)
            return True

        while not node.is_document_title and not node.is_root():
            if not node.paragraph.is_table():
                if node.paragraph.is_header():
                    break

                if float(node.paragraph.indentation) < float(
                        new_node.paragraph.indentation) - 1:
                    break

                if abs(node.paragraph.indentation - new_node.paragraph.indentation) < 1:
                    if node.paragraph.numbering:
                        break
                    node = node.parent
                    break

            node = node.parent

        if not node.can_has_child():
            node = node.parent

        node.add_child(new_node)

        return True
