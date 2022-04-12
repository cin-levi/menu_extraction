from . import BaseRule


class NumberingRule(BaseRule):
    def run(self, current_node, new_node) -> bool:
        numbering = new_node.paragraph.numbering
        if not numbering or numbering.is_initial():
            return False

        node = find_parent_with_numbering(current_node, new_node)

        if node.is_root() or node.is_document_title:
            return False

        node.add_child(new_node)

        return True


def find_parent_with_numbering(leaf, new_node):
    node = leaf
    numbering = new_node.paragraph.numbering
    if numbering.is_subsequence_of(leaf.paragraph.numbering):
        return leaf.parent

    while not node.is_root():
        if node.paragraph.numbering:
            if numbering.is_subsequence_of(node.paragraph.numbering):
                return node.parent

            if numbering.is_subsection_of(node.paragraph.numbering):
                return node

        node = node.parent

    return node
