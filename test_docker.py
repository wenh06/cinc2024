"""
"""

import os
from copy import deepcopy
from pathlib import Path
from typing import Union

import numpy as np
import torch
from torch.nn.parallel import DataParallel as DP  # noqa: F401
from torch.utils.data import DataLoader
from torch_ecg.utils.misc import dict_to_str, str2bool  # noqa: F401

from cfg import _BASE_DIR, ModelCfg, TrainCfg
from data_reader import CINC2024Reader
from dataset import CinC2024Dataset, collate_fn
from models import MultiHead_CINC2024
from utils.misc import func_indicator

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if ModelCfg.torch_dtype == torch.float64:
    torch.set_default_tensor_type(torch.DoubleTensor)
    DTYPE = np.float64
else:
    DTYPE = np.float32


tmp_data_dir = Path(os.environ.get("revenger_data_dir", _BASE_DIR / "tmp" / "CINC2024")).resolve()
print(f"tmp_data_dir: {str(tmp_data_dir)}")
tmp_data_dir.mkdir(parents=True, exist_ok=True)

dr = CINC2024Reader(tmp_data_dir)
# downloading is done outside the docker container
# and the data folder is mounted to the docker container as read-only
# dr.download()
# dr._ls_rec()


tmp_model_dir = Path(os.environ.get("revenger_model_dir", TrainCfg.model_dir)).resolve()

tmp_output_dir = Path(os.environ.get("revenger_output_dir", _BASE_DIR / "tmp" / "output")).resolve()


def echo_write_permission(folder: Union[str, Path]) -> None:
    is_writeable = "is writable" if os.access(str(folder), os.W_OK) else "is not writable"
    print(f"{str(folder)} {is_writeable}")


echo_write_permission(tmp_data_dir)


@func_indicator("testing dataset")
def test_dataset() -> None:
    """Test the dataset."""
    ds_config = deepcopy(TrainCfg)
    ds_config.db_dir = tmp_data_dir
    ds_config.working_dir = tmp_model_dir / "working_dir"
    ds_config.working_dir.mkdir(parents=True, exist_ok=True)

    ds_train = CinC2024Dataset(ds_config, training=True, lazy=True)
    ds_val = CinC2024Dataset(ds_config, training=False, lazy=True)

    assert len(ds_train) > 0
    assert len(ds_val) > 0

    # int indexing
    data = ds_val[0]
    assert isinstance(data, dict)
    assert "image" in data
    assert set(data.keys()) <= set(["dx", "digitization", "image", "mask"])
    # since the image backbone from transformers automatically converts the image to correct format
    # preprocessings including normalization, resizing, channel conversion, etc. are done in the backbone
    # instead of in the dataset
    assert isinstance(data["image"], np.ndarray)
    assert data["image"].ndim == 3
    assert data["image"].shape[-1] == 3
    if "dx" in data:
        assert isinstance(data["dx"], np.generic) and data["dx"].ndim == 0
    if "digitization" in data:
        assert isinstance(data["digitization"], np.ndarray)
        assert data["digitization"].ndim == 2
        assert data["digitization"].shape[0] == ds_val.config.num_leads
    if "mask" in data:
        assert isinstance(data["mask"], np.ndarray)
        assert data["mask"].ndim == 2
        assert data["mask"].shape[0] == ds_val.config.num_leads

    # slice indexing
    batch_size = 4
    data = ds_val[:batch_size]
    assert isinstance(data, dict)
    assert "image" in data
    assert set(data.keys()) <= set(["dx", "digitization", "image", "mask"])
    assert isinstance(data["image"], list)
    assert len(data["image"]) == batch_size
    assert all([isinstance(img, np.ndarray) for img in data["image"]])
    if "dx" in data:
        assert isinstance(data["dx"], torch.Tensor)
        assert data["dx"].ndim == 1
        assert data["dx"].shape[0] == batch_size
    if "digitization" in data:
        pass
        # assert isinstance(data["digitization"], torch.Tensor)
        # assert data["digitization"].ndim == 3
        # assert data["digitization"].shape[0] == batch_size
        # assert data["digitization"].shape[1] == ds_val.config.num_leads
    if "mask" in data:
        pass
        # assert isinstance(data["mask"], torch.Tensor)
        # assert data["mask"].ndim == 3
        # assert data["mask"].shape[0] == batch_size
        # assert data["mask"].shape[1] == ds_val.config.num_leads

    # DO NOT use the following code to load all data
    # since the dataset is too large to load all data into memory
    # ds_train._load_all_data()
    # ds_val._load_all_data()

    print("dataset test passed")


@func_indicator("testing models")
def test_models() -> None:
    """Test the models."""
    model = MultiHead_CINC2024()
    model.to(DEVICE)
    ds_config = deepcopy(TrainCfg)
    ds_config.db_dir = tmp_data_dir
    ds_val = CinC2024Dataset(ds_config, training=False, lazy=True)
    # ds_val._load_all_data()
    dl = DataLoader(
        dataset=ds_val,
        batch_size=24,
        shuffle=True,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_fn,
    )
    for idx, input_tensors in enumerate(dl):
        inference_output = model.inference(input_tensors["image"])
        print(inference_output)
        if idx > 5:
            break
    for idx, input_tensors in enumerate(dl):
        print(model.inference(input_tensors["image"]))
        if idx <= 5:
            continue
        input_images = input_tensors.pop("image")
        forward_output = model(model.get_input_tensors(input_images), labels=input_tensors)
        print(forward_output)
        if idx > 10:
            break

    print("models test passed")


@func_indicator("testing challenge metrics")
def test_challenge_metrics() -> None:
    """Test the challenge metrics."""
    # random prediction
    raise NotImplementedError("challenge metrics test is not implemented")

    print("challenge metrics test passed")


@func_indicator("testing trainer")
def test_trainer() -> None:
    """Test the trainer."""
    echo_write_permission(tmp_data_dir)
    echo_write_permission(tmp_model_dir)

    train_config = deepcopy(TrainCfg)
    train_config.db_dir = tmp_data_dir
    # train_config.model_dir = model_folder
    # train_config.final_model_filename = "final_model.pth.tar"
    train_config.debug = True
    train_config.working_dir = tmp_model_dir / "working_dir"
    train_config.working_dir.mkdir(parents=True, exist_ok=True)

    train_config.n_epochs = 5
    train_config.batch_size = 8  # 16G (Tesla T4)
    # train_config.log_step = 20
    # # train_config.max_lr = 1.5e-3
    # train_config.early_stopping.patience = 20

    print("trainer test is not implemented")
    return

    print("trainer test passed")


from evaluate_model import evaluate_model  # noqa: F401
from run_model import run_model  # noqa: F401

# from train_model import train_challenge_model
from team_code import train_challenge_model  # noqa: F401


@func_indicator("testing challenge entry")
def test_entry() -> None:
    """Test Challenge entry."""
    echo_write_permission(tmp_data_dir)
    echo_write_permission(tmp_model_dir)

    # run the model training function (script)
    print("run model training function")
    data_folder = tmp_data_dir
    train_challenge_model(str(data_folder), str(tmp_model_dir), verbose=2)

    # run the model inference function (script)
    output_dir = tmp_output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print("entry test is not implemented")
    return

    print("run model for the original data")

    # run_model(
    #     str(tmp_model_dir),
    #     str(data_folder),
    #     str(output_dir),
    #     allow_failures=False,
    #     verbose=2,
    # )

    print("evaluate model for the original data")

    # (
    #     challenge_score,
    #     auroc_outcomes,
    #     auprc_outcomes,
    #     accuracy_outcomes,
    #     f_measure_outcomes,
    #     mse_cpcs,
    #     mae_cpcs,
    # ) = evaluate_model(str(data_folder), str(output_dir))
    # eval_res = {
    #     "challenge_score": challenge_score,
    #     "auroc_outcomes": auroc_outcomes,
    #     "auprc_outcomes": auprc_outcomes,
    #     "accuracy_outcomes": accuracy_outcomes,
    #     "f_measure_outcomes": f_measure_outcomes,
    #     "mse_cpcs": mse_cpcs,
    #     "mae_cpcs": mae_cpcs,
    # }

    # print(f"Evaluation results: {dict_to_str(eval_res)}")

    print("entry test passed")


test_team_code = test_entry  # alias


if __name__ == "__main__":
    TEST_FLAG = os.environ.get("CINC2024_REVENGER_TEST", False)
    TEST_FLAG = str2bool(TEST_FLAG)
    if not TEST_FLAG:
        # raise RuntimeError(
        #     "please set CINC2024_REVENGER_TEST to true (1, y, yes, true, etc.) to run the test"
        # )
        print("Test is skipped.")
        print("Please set CINC2024_REVENGER_TEST to true (1, y, yes, true, etc.) to run the test")
        exit(0)

    print("#" * 80)
    print("testing team code")
    print("#" * 80)
    print(f"tmp_data_dir: {str(tmp_data_dir)}")
    print(f"tmp_model_dir: {str(tmp_model_dir)}")
    print(f"tmp_output_dir: {str(tmp_output_dir)}")
    print("#" * 80)

    test_dataset()
    test_models()
    # test_challenge_metrics()
    # test_trainer()  # directly run test_entry
    # test_entry()
    # set_entry_test_flag(False)
