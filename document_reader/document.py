import logging
from pathlib import Path
from typing import List, Generator

from .paragraph import Paragraph
from .utils.structuring.node import Node

logger = logging.getLogger(__name__)


class Document:
    def __init__(self, file_name: str or Path):
        self.file_path: Path = Path(file_name)
        paragraphs, imgs = self._load()
        self.paragraphs: List[Paragraph] = paragraphs
        self.imgs = imgs

    def __repr__(self):
        return f'{self.file_path.name}: {len(self.paragraphs)} paragraphs'

    def __iter__(self):
        yield from self.paragraphs

    @property
    def root(self) -> Node:
        return self._do_structuring()

    @property
    def ext(self) -> str:
        return Path(self.file_path).suffix.lower()

    def tree(self) -> Generator:
        return self.root.dfs()

    def to_json(self) -> str:
        from document_reader.utils import export
        return export(self)

    def to_xlsx(self) -> str:
        from .utils import export
        return export(self, '.xlsx')

    def _load_pdf(self):
        from .pdf import load_pdf
        pp, imgs = load_pdf(self.file_path)
        return pp, imgs

    def _load_unknown(self):
        raise NotImplementedError(
            f'Not supported extension: "{self.file_path.suffix.upper()}". '
            f'To convert to supported formats, including pdf and html,'
            f' see `utils.convert_file`'
        )

    def _load(self):
        return {'.pdf': self._load_pdf,
                }.get(self.ext, self._load_unknown)()

    def _do_structuring(self):
        from .utils import build_document_structure
        return build_document_structure(self.paragraphs)
