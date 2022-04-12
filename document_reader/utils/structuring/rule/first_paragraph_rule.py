from . import BaseRule


class FirstParagraphRule(BaseRule):
    """ Apply for the first paragraph only.
    Add the node, holding the first paragraph, to the root node.
    """

    def run(self, current_node, new_node) -> bool:
        if current_node.is_root():
            current_node.add_child(new_node)

            new_node.is_document_title = True
            return True

        return False
