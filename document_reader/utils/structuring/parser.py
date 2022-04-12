import logging

from .node import RootNode, Node
from ..utils import show_tree

logger = logging.getLogger(__name__)


def build_document_structure(paragraphs):
    logger.info(f'Building document structure')

    root = RootNode()
    current_node = root
    for index, paragraph in enumerate(paragraphs):
        node = Node(paragraph, index)
        from ..config import settings
        for rule in settings.pdf.structuring.rules:
            try:
                if rule.run(current_node, node):
                    current_node = node
                    current_node.rule_name = rule.name
                    break
            except BaseException:
                logger.exception(f'Structure parsing exception!')
                raise

    show_tree(root)
    return root
