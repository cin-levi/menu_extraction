from typing import Optional, List

from ...paragraph import Paragraph


class Node:
    def __init__(self, paragraph: Paragraph = None, index: int = 0):
        """ A node of bidirectional tree, representing a document structure.
        """
        self.paragraph: Paragraph = paragraph
        self.index: int = index

        self.parent: Optional[Node] = None
        self.children: List[Node] = []
        self.is_document_title = False

        self.rule_name: str = ''

    def __repr__(self):
        return f'{self.index}: {self.text}'

    @property
    def page_number(self):
        return self.paragraph.page_number

    @property
    def text(self):
        return self.paragraph.text

    @property
    def title(self):
        return self.paragraph.title

    @property
    def is_header(self):
        return self.paragraph.is_header() or len(self.children) > 0

    @property
    def is_table(self):
        return self.paragraph.is_table()

    @property
    def parent_index(self) -> Optional[int]:
        if not self.parent.is_root():
            return self.parent.index

    @property
    def depth(self):
        return len(list(self.up()))

    def can_has_child(self) -> bool:
        return any([
            self.is_root(),
            self.is_document_title,
            self.paragraph.is_header(),
            self.paragraph.numbering,
            self.paragraph.bullet,
        ])

    def is_root(self) -> bool:
        return self.parent is None

    def add_child(self, child):
        child.parent = self
        self.children.append(child)

    def root(self):
        return self if self.is_root() else self.parent.root()

    def up(self):
        yield self
        if not self.is_root():
            yield from self.parent.up()

    def dfs(self):
        """ Traveling the tree which root is self, in DFS(Depth First Search)
        preordering.
        """
        yield self
        for child in self.children:
            yield from list(child.dfs())


class RootNode(Node):
    def __init__(self):
        super().__init__()

    def __repr__(self):
        return 'root'

    def dfs(self):
        for child in self.children:
            yield from list(child.dfs())
