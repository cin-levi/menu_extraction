from pathlib import Path
from typing import List

from pdfplumber.page import Page
from pdfplumber.utils import to_list, cluster_objects


def get_rows(cells):
    import itertools
    from operator import itemgetter

    def get_centroids(indices):
        vertices = list(
            sorted(set(itertools.chain(*map(itemgetter(*indices), cells)))))
        for index in range(1, len(vertices)):
            yield (vertices[index - 1] + vertices[index]) / 2

    x_centroids = list(get_centroids((0, 2)))
    y_centroids = list(get_centroids((1, 3)))

    for y in y_centroids:
        yield [
            list(filter(lambda c: c[0] < x < c[2] and c[1] < y < c[3], cells))[
                0]
            for x in x_centroids]


def filter_pages(pages: List[Page]):
    from ...utils.config import settings
    allows_pages = settings.debug.pdf.pages
    if allows_pages:
        return filter(lambda page: page.page_number in allows_pages, pages)
    return pages


def is_contain(container, containee):
    x0, y0, x1, y1 = container
    x0_, y0_, x1_, y1_ = containee
    return \
        x0 <= (x0_ + x1_) / 2 <= x1 and \
        y0 <= (y0_ + y1_) / 2 <= y1


def extract_lines(chars) -> List[List]:
    from document_reader import settings
    chars = to_list(chars)
    y_clusters = cluster_objects(chars, "doctop", settings.pdf.tolerance_y)
    clusters = []
    for cluster in y_clusters:
        clusters.extend(cluster_objects(cluster, 'x0', settings.pdf.tolerance_x))
    return clusters


def guess_alignment(page_rect, text_rect):
    from document_reader.pdf.style import ALIGNMENT
    page_width = page_rect['x1'] - page_rect['x0']

    left_margin = text_rect['x0'] - page_rect['x0']
    right_margin = page_rect['x1'] - text_rect['x1']

    if abs(left_margin - right_margin) < ALIGNMENT.THRESHOLD:
        return ALIGNMENT.CENTER

    if left_margin > page_width / 2:
        return ALIGNMENT.RIGHT

    if right_margin > page_width / 2:
        return ALIGNMENT.LEFT

    return ALIGNMENT.UNKNOWN
