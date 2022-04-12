from . import BaseRule


class BulletRule(BaseRule):
    """Handle node which has paragraph starts with a bullet
    All consecutive paragraphs start with the same bullet(symbol + indentation) must be sibling.
    """

    def run(self, current_node, new_node) -> bool:
        node = current_node
        bullet_1 = new_node.paragraph.bullet
        bullet_2 = current_node.paragraph.bullet
        if bullet_1 and bullet_2 and bullet_1 == bullet_2:
            if node.paragraph.numbering:
                node.add_child(new_node)
            else:
                node.parent.add_child(new_node)

            return True

        return False
