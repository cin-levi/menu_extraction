from .utils import *


class Experiment:
    path = Path('.experiment')
    artifact_path = path / 'artifact'
    latest_exp_path = path / 'latest'
    latest_artifact_path = latest_exp_path / 'artifact'

    def init(self):
        self.artifact_path.mkdir(parents=True, exist_ok=True)

        shutil.rmtree(self.latest_exp_path, ignore_errors=True)
        self.latest_exp_path.mkdir()

    @validate_arguments
    def register(self, metric: str, filename: Path, result: float):
        with open(self.latest_exp_path.joinpath(f'{metric}.csv'), mode='a',
                  encoding='utf8') as f:
            csv.writer(f).writerow((filename.name, round(result, 2)))

    @validate_arguments
    def set_artifact(self, artifact_path: Path):
        shutil.copytree(src=artifact_path, dst=self.latest_artifact_path)

    def finish(self):
        """ Update metric with the latest experiment
        """
        id_ = append_latest_metrics(self.latest_exp_path.glob('*.csv'))
        copy_artifact(self.latest_artifact_path, id_)

    def report(self, metric='tree', max_records=0):
        rows = read_csv(self.path / f'{metric}.csv')
        rows = {k: rows[k] for k in rows if rows[k][-1]}

        delete_column_with_same_result(rows)
        limit_total_number_of_record(rows, max_records)

        pretty_print_csv(rows, metric)

    def diff(self, filename: str, id_: int, id_base: int = None):
        from document_reader.utils.evalutating import gen_report
        from document_reader.utils.utils import launch
        id_ = id_ or get_latest_run_id()
        id_base = id_base or (id_ - 1)
        dr = self.artifact_path / str(id_) / f'{filename}_dr.xlsx'
        ca = self.artifact_path / str(id_base) / f'{filename}_dr.xlsx'
        out_dir = Path('.tmp')
        gen_report(dr=dr, ca=ca, out_dir='.tmp', register_output=False)
        launch(out_dir / f'{filename}_dr_report.xlsx')

    def reset(self):
        confirmed = input(
            'Delete all the experiment records (y/n)?').upper() == 'Y'
        if confirmed:
            shutil.rmtree(self.path)
        else:
            print('Cancelled!')


exp = Experiment()
