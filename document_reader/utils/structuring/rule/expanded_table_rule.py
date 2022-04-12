import logging

from . import BaseRule

logger = logging.getLogger(__name__)


class ExpandedTableRule(BaseRule):

    def run(self, current_node, new_node) -> bool:
        from document_reader.pdf.table import ExpandedTableRow
        if isinstance(new_node.paragraph, ExpandedTableRow) \
                and isinstance(current_node.paragraph, ExpandedTableRow):
            column_index = new_node.paragraph.column_index
            if current_node.paragraph.column_index < column_index:
                current_node.add_child(new_node)
                return True

            from document_reader import settings
            rules = settings.pdf.structuring.rules
            for rule in rules[rules.index(self) + 2:]:
                try:
                    if rule.run(current_node, new_node):
                        parent = new_node.parent.paragraph
                        if isinstance(parent,
                                      ExpandedTableRow) and column_index - parent.column_index in (
                        0, 1):
                            return True
                        else:
                            new_node.parent.children.remove(new_node)
                except BaseException as e:
                    logger.warning(f'Expanded table rule: {e}')

            for node in current_node.up():
                if isinstance(node, ExpandedTableRow) and \
                        node.column_index < column_index:
                    node.add_child(new_node)
                    return True

        return False
