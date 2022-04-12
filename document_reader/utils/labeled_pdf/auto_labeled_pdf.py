import logging
from pathlib import Path
from typing import List

from pdf_annotate import PdfAnnotator, Location, Appearance, Metadata
from pydantic import validate_arguments

logger = logging.getLogger(__name__)


class AnnotatedParagraph:
    def __init__(self, node):
        self.node = node

    @property
    def note(self):
        if self.node.is_table:
            return 'TBR'
        if self.node.children:
            return f'H{self.node.depth - 2}'
        return ''

    @property
    def annotation_data(self):
        BULGARIAN_ROSE = (0.30, 0.00, 0.05)
        TRACTOR_RED = (1.00, 0.10, 0.25)
        APPLE_GREEN = (0.67, 0.80, 0.00)

        r = self.node.paragraph.rect
        page_height = float(self.node.paragraph.page.height)
        # noinspection PyArgumentList
        location = Location(
            x1=float(r['x0']),
            y1=page_height - float(r['bottom']),
            x2=float(r['x1']),
            y2=page_height - float(r['top']),
            page=self.node.page_number - 1
        )
        if self.note.startswith('H'):
            stroke_color = TRACTOR_RED
        elif self.note.startswith('TB'):
            stroke_color = APPLE_GREEN
        else:
            stroke_color = BULGARIAN_ROSE

        # noinspection PyArgumentList
        return ('square',
                location,
                Appearance(stroke_color=stroke_color,
                           stroke_width=1),
                Metadata(Contents=self.note))


class AutoLabeledPDF:
    def __init__(self, path: str or Path):
        from document_reader import Document
        self.document = Document(path)

    def save(self):
        logger.info(f'Saving {self.document.file_path.name}')

    @validate_arguments
    def save_as(self, out: Path):
        logger.info(f'Saving labeled pdf as {out}')
        out.parent.mkdir(exist_ok=True, parents=True)

        annotator = PdfAnnotator(str(self.document.file_path))
        for node in self.document.tree():
            annotator.add_annotation(*AnnotatedParagraph(node).annotation_data)

        annotator.write(str(out))  # or use overwrite=True if you feel luc
        from document_reader.utils.utils import launch
        launch(out)

    @property
    def annotated_paragraph(self) -> List[AnnotatedParagraph]:
        return list(map(AnnotatedParagraph, self.document.tree()))
