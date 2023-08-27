import copy
from pathlib import Path

import pandas as pd
import pytest

from ablator import PlotAnalysis
from ablator.analysis.results import Results


def get_best(x: pd.DataFrame, task_type: str):
    if task_type == "regression":
        return x.sort_values("val_rmse", na_position="last").iloc[0]
    else:
        return x.sort_values("val_acc", na_position="first").iloc[-1]


@pytest.mark.order(1)
def test_analysis(tmp_path: Path, ablator_results):
    results: Results = ablator_results
    PlotAnalysis(results, optim_metrics={"val_loss": "min"})
    categorical_name_remap = {
        "model_config.mock_param": "Some Parameter",
    }
    numerical_name_remap = {
        "train_config.optimizer_config.arguments.lr": "Learning Rate",
    }
    analysis = PlotAnalysis(
        results,
        save_dir=tmp_path.as_posix(),
        cache=True,
        optim_metrics={"val_loss": "min"},
    )
    attribute_name_remap = {**categorical_name_remap, **numerical_name_remap}
    analysis.make_figures(
        metric_name_remap={
            "val_loss": "Val. Loss",
        },
        attribute_name_remap=attribute_name_remap,
    )
    assert all(
        tmp_path.joinpath("violinplot", "val_loss", f"{file_name}.png").exists()
        for file_name in categorical_name_remap
    )

    assert all(
        tmp_path.joinpath("linearplot", "val_loss", f"{file_name}.png").exists()
        for file_name in numerical_name_remap
    )


if __name__ == "__main__":
    import copy
    import shutil

    from tests.conftest import DockerRayCluster, run_tests_local
    from tests.test_plugins.model import WORKING_DIR, get_ablator

    l = locals()
    fn_names = [fn for fn in l if fn.startswith("test_")]
    test_fns = [l[fn] for fn in fn_names]
    ablator_tmp_path = Path("/tmp/ablator_tmp")
    shutil.rmtree(ablator_tmp_path, ignore_errors=True)
    ray_cluster = DockerRayCluster(working_dir=WORKING_DIR)
    ray_cluster.setUp()

    ablator = get_ablator(
        ablator_tmp_path,
        working_dir=Path(WORKING_DIR).parent,
        main_ray_cluster=ray_cluster,
    )
    config = ablator.run_config
    ablator_results = Results(config, ablator.experiment_dir)
    kwargs = {
        "ablator_results": copy.deepcopy(ablator_results),
    }
    run_tests_local(test_fns, kwargs)
