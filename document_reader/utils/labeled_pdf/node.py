import logging

from ..structuring.node import Node as BaseNode

logger = logging.getLogger(__name__)


class Node(BaseNode):
    def __init__(self, box=None, index: int = 0):
        """ A node of bidirectional tree, representing a document structure.
        """
        super().__init__(box, index)
        self.box = box

    @property
    def note(self):
        return self.box.note

    @property
    def is_header(self):
        return self.note.startswith('H') or self.note == 'DT'

    @property
    def is_table(self):
        return self.note.startswith('T')

    def add_new_leaf(self, leaf):
        PARENT_NOTE = dict(
            DT='root',
            H1='DT', H2='H1', H3='H2', H4='H3', H5='H4',
            H6='H5', H7='H6', H8='H7', H9='H8', H10='H9',
            P='H', T='H', TH='H', TV='H',
        )
        try:
            self.find_up(PARENT_NOTE[leaf.box.note]).add_child(leaf)
        except KeyError:
            raise AssertionError(
                f'L05: Cannot parse annotation: {leaf.box.note}')

        return leaf

    def find_up(self, note: str):
        if note == 'root':
            return self.root()

        if self.is_root() or self.box.note == 'DT':
            return self

        if self.box.note.startswith(note):
            return self

        return self.parent.find_up(note)


class RootNode(Node):
    def __init__(self):
        super().__init__()

    def __repr__(self):
        return 'root'

    def dfs(self):
        for child in self.children:
            yield from list(child.dfs())
