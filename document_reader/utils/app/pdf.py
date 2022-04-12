import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def _try_import_pikepdf():
    try:
        import pikepdf
    except ImportError:
        raise ImportError('Failed to import `pikepdf`. '
                          'Run `pip3 install pikepdf` for installing')


class PDF:
    def __init__(self, path: str):
        _try_import_pikepdf()
        self.path = Path(path)

    def split_pages(self, page_range) -> Path:
        from pikepdf import Pdf
        src = Pdf.open(self.path)
        dst = Pdf.new()

        length = len(src.pages)
        page_range = [i if i >= 0 else length + i for i in page_range]
        for index, page in enumerate(src.pages):
            if index in page_range:
                dst.pages.append(page)

        out_file = self.path.with_suffix(
            f'.page_{"-".join(map(str, page_range))}.pdf')
        dst.save(out_file)

        logger.info(f'Split pdf: {out_file.relative_to(".")}')
        return out_file

    def remove_password(self,
                        out_dir: str = '.password_removed_pdf'
                        ) -> Path:
        import pikepdf

        out_dir = Path(out_dir)
        out_dir.mkdir(exist_ok=True, parents=True)
        out_file = out_dir / self.path.name

        pdf = pikepdf.open(self.path)
        pdf.save(out_file)

        logger.info(f'Removed password PDF: {out_file.relative_to(".")}')

        return out_file
