import logging
from pathlib import Path
from typing import Generator, Optional

from .utils import export_to_xlsx, load_labeled_pdf, export_to_json

logger = logging.getLogger(__name__)


class LabeledPDF:
    def __init__(self, file_path: str or Path):
        """ LabeledPDF is a class representing a pdf labeled with a specific
         annotation convention
        :param file_path: path to an annotated pdf
        """
        self.file_path = Path(file_path)
        self.root = load_labeled_pdf(self.file_path)

    def __iter__(self):
        yield from self.boxes

    @property
    def boxes(self):
        return [node.box for node in self.tree()]

    @property
    def doc_tree(self):
        return self.root

    def tree(self) -> Generator:
        return self.doc_tree.dfs()

    def to_xlsx(self, out_dir: Optional[str or Path] = None):
        """ Export the labeled pdf to the standard xlsx output format.
        :param out_dir: [Optional] path to a directory stored the output files,
        default: same name, with .xlsx extension, placed next the pdf file
        """
        xlsx_file = ((out_dir or self.file_path.parent) / self.file_path.name). \
            with_suffix('.xlsx')

        logger.info(f'Exporting labeled pdf to {xlsx_file}')
        return export_to_xlsx(self.doc_tree, xlsx_file)

    def to_json(self, out_dir: Optional[str or Path] = None):
        """ Export the labeled pdf to json
        :param out_dir: [Optional] path to a directory stored the output files,
        default: same directory as in the input files
        :return: path to the output file
        """
        json_file = ((out_dir or self.file_path.parent) / self.file_path.name). \
            with_suffix('.json')

        logger.info(f'Exporting labeled pdf to {json_file}')
        return export_to_json(self.doc_tree, json_file)
