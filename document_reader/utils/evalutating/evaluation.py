import logging
from multiprocessing import Pool, cpu_count
from pathlib import Path

from .document import Document
from .utils import write_report, map_with_ca, correct_ocr_error, \
    write_accumulated_report

logger = logging.getLogger(__name__)


def gen_report(dr: Path or str, ca: Path or str,
               out_dir='.report', jobs: int = None,
               register_output: bool = True):
    """ Generate report base on `ca`, and the output from lib-document-reader `dr` for short.
    :param register_output: whether register the output to the experiment management tool, default: True
    :param dr: a path to a file or directory holding the output of lib-document-reader
    :param ca: a path to a file or a directory holding the correct answer, or the ground truth
    :param out_dir: a path to a directory where the to-be-generated report will be placed, default: ".report"
    :param jobs: number of jobs run in parallel, default: None (=number of the CPU)
    """
    mapping = map_with_ca(Path(dr), Path(ca))
    if jobs == 1:
        reports = [_make_report(dr, ca, out_dir) for dr, ca in mapping]
    else:
        with Pool(cpu_count() if jobs is None else jobs) as p:
            reports = p.starmap(
                _make_report,
                ((dr, ca, out_dir) for dr, ca in mapping))

    write_accumulated_report(reports, out_dir, register_output)


def _make_report(dr: Path, ca: Path, output_dir='.report') -> dict:
    logger.info(f'Comparing "{dr.name}" <---> "{ca.name}"')

    doc_dr, doc_ca = map(Document, (dr, ca))
    correct_ocr_error(doc_ca, doc_dr)
    return write_report(dict(
        dr=doc_dr,
        ca=doc_ca,
        out_dir=Path(output_dir)
    ))
