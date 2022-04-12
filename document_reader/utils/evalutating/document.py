from pathlib import Path
from typing import List

from document_reader.utils.utils import normalize_text


class Paragraph:
    def __init__(self, page_number=0, text='', index='', parent_index=''):
        self.page_number = page_number
        self.text: str = text
        self.index: str = index
        self.parent_index: str = parent_index
        self.corrected_text = None

    def __str__(self):
        return self.str()

    def __repr__(self):
        return self.str()

    def str(self):
        return self.text if self.corrected_text is None else self.corrected_text


class Document:
    def __init__(self, file_path: str or Path):
        self.file: Path = Path(file_path)
        self.paragraphs = self._load()
        self.corrected_text = None

    def __repr__(self):
        return f'"{self.file.name}" - a document with {len(self.paragraphs)} paragraphs.'

    def __iter__(self) -> Paragraph:
        yield from self.paragraphs

    def _load(self) -> List[Paragraph]:
        file_extension = self.file.suffix.lower()
        if file_extension == '.xlsx':
            return self._load_excel()

        raise NotImplementedError(f'"{file_extension}" file extension has not been supported.')

    def _load_excel(self, headers=('Page Number', 'Text', 'Index', 'Parent Index')):
        from pandas import read_excel
        rows = read_excel(self.file, dtype=str, engine='openpyxl').loc[:, headers].fillna('').values

        return [Paragraph(*map(normalize_text, row)) for row in rows]

    @property
    def text(self) -> str:
        return '\n'.join(map(lambda p: normalize_text(p.text, remove_space=True), self.paragraphs))

    def get(self, index):
        if str(index):
            for p in self.paragraphs:
                if p.index == index:
                    return p
