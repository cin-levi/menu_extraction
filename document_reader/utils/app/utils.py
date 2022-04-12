import logging
import shutil
import time
from functools import partial
from multiprocessing import cpu_count, Pool
from pathlib import Path
from typing import List, Optional

from .. import convert_file
from ..config import settings
from ..labeled_pdf import LabeledPDF
from ..labeled_pdf import validate_labeled_pdf
from ..utils import launch

logger = logging.getLogger(__name__)


def _dummy_fn():
    pass


activate_settings_fn = _dummy_fn


def _process_file(filename: Path):
    logger.info('PROCESSING: ' + filename.name)

    activate_settings_fn()

    tic = time.perf_counter()
    if filename.suffix in ('.doc', '.docx'):
        filename = convert_file(filename, Path(filename).with_suffix('.pdf'))

    from document_reader import Document
    doc = Document(filename)

    outfile = doc.to_json()

    toc = time.perf_counter()
    logger.debug(f'Finish {Path(filename).name} in {toc - tic:0.2f} seconds')

    return outfile


def _process_folder(path: Path, allowed_extensions, job=None):
    tic = time.perf_counter()
    logger.info('PROCESS FOLDER: ' + path.name)
    files = [f for f in path.rglob('*.*') if f.suffix in allowed_extensions]
    if job == 1:
        for index, file in enumerate(files):
            try:
                logger.info(f'{index}/{len(files)} - {file}')
                _process_file(file)
            except BaseException:
                logger.exception('')
                raise
    else:
        job = job or cpu_count()
        with Pool(job) as p:
            p.map(_process_file, files)

    toc = time.perf_counter()
    logger.debug(f'Finish in {toc - tic:0.2f} seconds')


def run(path: str, pages=None, show_tree=False, show_para=False, job=None,
        gen_image=False, allowed_extensions=('.pdf', '.html')):
    """ Run document reader against a file or directory specified by `path`
    :param path: path to a file or directory
    :param pages: specify which pages to be processed, default: all pages
    :param show_tree: set to True to show output document tree for debugging, default: False
    :param show_para: set to True to show constructed paragraph for debugging, default: False
    :param job: number of parallel jobs(process) to be run at the same time, default:=number of cpu
    :param gen_image: set to True to generate debugging image
    :param allowed_extensions: a whitelist of extensions, default: .pdf and .html
    """
    path = Path(path)
    assert path.exists(), f'Path not exist {path}'

    # >> Settings
    settings.debug.show_para = show_para
    settings.debug.show_tree = show_tree

    settings.debug.pdf.pages = pages or []
    settings.debug.pdf.gen_image = gen_image

    import shutil
    shutil.rmtree(settings.exporting.location, ignore_errors=True)
    shutil.rmtree(settings.debug.location, ignore_errors=True)

    if path.is_file():
        _process_file(path)
    else:
        _process_folder(path, allowed_extensions, job)


def gen_report(ca_dir: str or Path):
    """ Generate detail report for evaluating the output of document reader.
    Can be used to evaluate and to do failure analysis.
    :param ca_dir: path to the ground truth 
    """
    from document_reader.utils.evalutating import gen_report
    shutil.rmtree('.report', ignore_errors=True)
    gen_report(settings.exporting.location, ca_dir)


def show_statistic(metrics=('tree', 'para', 'ocr',)):
    from document_reader.utils.managing_experiment import exp
    if isinstance(metrics, str):
        metrics = [metrics, ]

    for metric in metrics:
        exp.report(metric=metric)


def show_diff(filename: str, run_id_1: int = None, run_id_2: int = None):
    from document_reader.utils.managing_experiment import exp
    exp.diff(filename, run_id_1, run_id_2)


def open_converted_labeled_pdf(file_path: str = 'data/labeled_pdf/ok.pdf'):
    launch(LabeledPDF(file_path).to_xlsx())


def open_report(file_name: str):
    launch(f'.report/{file_name}_report.xlsx')


def open_debug_image(file_name: str, page_number: int):
    launch(f'{settings.debug.location}/{file_name}-page-{page_number}.png')


def collect_files(path: str or Path,
                  recursive=False,
                  suffices=('.pdf',)) -> List[Path]:
    suffices = {s.lower() for s in suffices}
    path = Path(path)
    if path.is_file():
        return [path] if path.suffix in suffices else []

    glob_fn = path.rglob if recursive else path.glob
    # noinspection PyArgumentList
    return [file for file in glob_fn('*.*') if file.suffix.lower() in suffices]


def auto_label_pdf(path: str or Path, out_dir: Path = None):
    from ..labeled_pdf.auto_labeled_pdf import AutoLabeledPDF
    import shutil
    out_dir = Path(out_dir or settings.labeling.location)
    shutil.rmtree(out_dir, ignore_errors=True)
    for file in collect_files(path):
        try:
            logger.info(f'Run auto labeling for "{file.name}"')
            AutoLabeledPDF(file).save_as(out_dir / file.name)
        except BaseException:
            logger.exception(f'Failed to run {file}')


def convert_labeled_pdf(path: str, out_format: str,
                        out_dir: Optional[str] = None):
    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(exist_ok=True, parents=True)

    for file in collect_files(path):
        if out_format == '.xlsx':
            LabeledPDF(file).to_xlsx(out_dir)
        elif out_format == '.json':
            LabeledPDF(file).to_json(out_dir)
        else:
            raise ValueError(f'Unsupported format `{out_format}`')


convert_labeled_pdf_to_xlsx = partial(convert_labeled_pdf, out_format='.xlsx')
convert_labeled_pdf_to_json = partial(convert_labeled_pdf, out_format='.json')

__all__ = [
    'activate_settings_fn',
    'auto_label_pdf',
    'convert_file',
    'gen_report',
    'LabeledPDF',
    'launch',
    'open_converted_labeled_pdf',
    'open_debug_image',
    'open_report',
    'run',
    'settings',
    'show_diff',
    'show_statistic',
    'validate_labeled_pdf',
    'convert_labeled_pdf_to_xlsx',
    'convert_labeled_pdf_to_json',
]
