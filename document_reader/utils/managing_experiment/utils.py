import csv
import shutil
from operator import truth
from pathlib import Path
from typing import List

from colorama import Back
from pydantic import validate_arguments


@validate_arguments()
def read_csv(path: Path):
    assert path.exists(), f'File not exist {path}'

    with open(path, encoding='utf8') as f:
        return {r[0].lstrip('*'): r[1:] for r in csv.reader(f)}


@validate_arguments
def csv_dict_to_list(dict_rows: dict):
    rows = [['id'] + dict_rows['id']]
    rows += [[k] + dict_rows[k] for k in sorted(dict_rows.keys()) if k != 'id']
    return rows


@validate_arguments
def write_csv(path: Path, results: dict):
    with open(path, mode='w', encoding='utf8') as f:
        csv.writer(f).writerows(csv_dict_to_list(results))


@validate_arguments
def copy_artifact(path: Path, id_: int):
    from . import exp
    shutil.copytree(src=path, dst=exp.artifact_path / str(id_))


def append_latest_metric(metric: Path):
    from . import exp
    path = exp.path / metric.name
    rows = read_csv(path) if path.exists() else {}
    latest = read_csv(metric)

    latest['id'] = [int(rows['id'][-1] if rows else 0) + 1]

    if rows:
        for k, v in rows.items():
            v.extend(latest[k] if k in latest else [None])
        for k, v in latest.items():
            if k not in rows:
                rows[f'*{k}'] = [None] * (len(rows['id']) - 1) + v
    else:
        rows = latest

    write_csv(path, rows)
    return latest['id'][0]


def find_last_result(results: List, start_index: int):
    for x in results[start_index:]:
        if x:
            return x


def append_latest_metrics(metric_files) -> int:
    id_ = 0
    for metric in metric_files:
        id_ = append_latest_metric(metric)
    return id_


@validate_arguments()
def pretty_print_csv(dict_rows: dict, metric: str):
    from colorama import init
    init(autoreset=True)
    rows = [['id'] + list(reversed(dict_rows['id']))]
    rows += [[k] + list(reversed(dict_rows[k])) for k in
             sorted(dict_rows.keys()) if k != 'id']

    def delta(r: List, i: int):
        a = r[i]
        b = find_last_result(r, i + 1)
        if a and b:
            return float(a) - float(b)

        return 0

    def to_float(r, i):
        value = r[i]
        if value is None:
            value = find_last_result(r, i + 1)

        return float(value) if value else 0

    footer = []
    for index in range(1, len(rows[0])):
        n = sum(map(lambda r: truth(r[index] != ''), rows[1:]))
        diff = None
        if index < len(rows[0]) - 1:
            diff = round(sum(delta(r, index) for r in rows[1:]) / n, 2)

        total = round(sum(to_float(r, index) for r in rows[1:]) / n, 2)
        footer.append((diff, total))
    else:
        # noinspection PyTypeChecker
        rows.append(['TOTAL', ] + [r[1] for r in footer])
        # noinspection PyTypeChecker
        rows.append(['DELTA', ] + [r[0] for r in footer])

    w = str(max(len(row[0]) for row in rows) + 1)
    wh = str(int(w) - len(metric))
    normal = '{:>' + w + '} ' + '| {:<5} ' * (len(rows[0]) - 1)
    header_style = Back.LIGHTGREEN_EX + metric.upper() + Back.GREEN
    header = header_style + '{:>' + wh + '} ' + '| {:5} ' * (len(rows[0]) - 1)
    for index, row in enumerate(rows):
        fmt = header
        if index:
            fmt = (Back.BLACK if index % 2 else Back.LIGHTBLACK_EX) + normal
        print(fmt.format(*map(lambda x: '-  ' if x is None else x, row)))
    else:
        print('')


def delete_column_with_same_result(rows):
    def find(vector, index):
        for x in reversed(vector[:index]):
            if x:
                return x

    def remove():
        for key, value in rows.items():
            for index, x in enumerate(value):
                if x and x == find(value, index):
                    rows[key][index] = None
        values = [v for k, v in rows.items() if k != 'id']
        indices = [
            index
            for index in range(len(rows['id']) - 1)
            if not any(v[index] for v in values)
        ]
        is_changed = False
        for key in rows:
            for index in reversed(indices):
                del rows[key][index]
                is_changed = True
        return is_changed

    while remove():
        pass


@validate_arguments()
def limit_total_number_of_record(rows: dict, max_records: int):
    if max_records == 0:
        return rows

    for k, v in rows.items():
        index = max(0, len(rows[k]) - max_records)
        offset = 1
        while v[index] is None:
            rows[k][index] = v[index - offset]
            offset += 1

        rows[k] = rows[k][index:]


def get_latest_run_id():
    from .experiemnt import Experiment
    return max([int(dir.name) for dir in Experiment.artifact_path.glob(r'*')])
