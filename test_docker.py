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
from torch_ecg.cfg import CFG
from torch_ecg.components.metrics import ClassificationMetrics
from torch_ecg.utils.misc import dict_to_str, str2bool  # noqa: F401

from cfg import _BASE_DIR, ModelCfg, TrainCfg
from data_reader import CINC2024Reader
from dataset import CinC2024Dataset, collate_fn
from evaluate_model import run as model_evaluator_func
from models import MultiHead_CINC2024
from outputs import CINC2024Outputs
from run_model import run as model_runner_func
from team_code import REMOTE_HEADS_URL, REMOTE_HEADS_URL_ALT, SYNTHETIC_IMAGE_DIR, train_digitization_model, train_dx_model
from trainer import CINC2024Trainer
from utils.misc import func_indicator, url_is_reachable
from utils.scoring_metrics import compute_challenge_metrics, compute_digitization_metrics, compute_dx_metrics

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if ModelCfg.torch_dtype == torch.float64:
    torch.set_default_tensor_type(torch.DoubleTensor)
    DTYPE = np.float64
else:
    DTYPE = np.float32


tmp_data_dir = Path(os.environ.get("revenger_data_dir", _BASE_DIR / "tmp" / "CINC2024")).resolve()
print(f"tmp_data_dir: {str(tmp_data_dir)}")
# create the data directory if it does not exist
tmp_data_dir.mkdir(parents=True, exist_ok=True)
# list files and folders in the data directory
print(os.listdir(tmp_data_dir))

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
echo_write_permission(tmp_model_dir)
echo_write_permission(tmp_output_dir)


@func_indicator("testing dataset")
def test_dataset() -> None:
    """Test the dataset."""
    ds_config = deepcopy(TrainCfg)
    ds_config.db_dir = tmp_data_dir
    ds_config.working_dir = tmp_model_dir / "working_dir"
    ds_config.working_dir.mkdir(parents=True, exist_ok=True)

    echo_write_permission(tmp_data_dir)
    echo_write_permission(tmp_model_dir)
    echo_write_permission(ds_config.working_dir)

    ds_train = CinC2024Dataset(ds_config, training=True, lazy=True)
    ds_val = CinC2024Dataset(ds_config, training=False, lazy=True)

    print(f"{len(ds_train) = }, {len(ds_val) = }")
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
    echo_write_permission(tmp_data_dir)
    echo_write_permission(tmp_model_dir)
    echo_write_permission(tmp_output_dir)

    model = MultiHead_CINC2024()
    model.to(DEVICE)
    ds_config = deepcopy(TrainCfg)
    ds_config.db_dir = tmp_data_dir
    ds_val = CinC2024Dataset(ds_config, training=False, lazy=True)
    # ds_val._load_all_data()
    dl = DataLoader(
        dataset=ds_val,
        batch_size=4,
        shuffle=True,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_fn,
    )
    for idx, input_tensors in enumerate(dl):
        inference_output = model.inference(input_tensors["image"])
        print(inference_output)
        if idx > 1:
            break
    for idx, input_tensors in enumerate(dl):
        print(model.inference(input_tensors["image"]))
        if idx <= 1:
            continue
        input_images = input_tensors.pop("image")
        forward_output = model(model.get_input_tensors(input_images), labels=input_tensors)
        print(forward_output)
        if idx > 2:
            break

    # test classmethod "from_remote_heads"
    if not url_is_reachable("https://www.dropbox.com/"):
        remote_heads_url = REMOTE_HEADS_URL_ALT
    else:
        remote_heads_url = REMOTE_HEADS_URL
    model = MultiHead_CINC2024.from_remote_heads(
        url=remote_heads_url,
        model_dir=tmp_model_dir,
        device=DEVICE,
    )
    for idx, input_tensors in enumerate(dl):
        print(model.inference(input_tensors["image"]))
        if idx > 2:
            break

    print("models test passed")


@func_indicator("testing challenge metrics")
def test_challenge_metrics() -> None:
    """Test the challenge metrics."""
    # test compute_dx_metrics
    labels = [{"dx": ["Abnormal", "Normal", "Normal"]}, {"dx": ["Normal", "Normal"]}]
    outputs = [
        CINC2024Outputs(dx=["Abnormal", "Normal", "Abnormal"], dx_classes=["Abnormal", "Normal"]),
        CINC2024Outputs(dx=["Abnormal", "Normal"], dx_classes=["Abnormal", "Normal"]),
    ]
    assert compute_dx_metrics(labels, outputs) == {"f_measure": 0.5833333333333333}

    cm = ClassificationMetrics(multi_label=False)
    cm(labels=np.array([0, 1, 1, 1, 1]), outputs=np.array([0, 1, 0, 0, 1]), num_classes=2)
    assert cm.f1_measure == compute_dx_metrics(labels, outputs)["f_measure"]

    # test compute_digitization_metrics
    # TODO: implement the test, using labels and outputs containing digitization fields
    compute_digitization_metrics(labels, outputs)  # currently all nan

    # test compute_challenge_metrics
    assert compute_challenge_metrics(labels, outputs) == {
        "dx_f_measure": 0.5833333333333333,
        "digitization_snr": np.nan,
        "digitization_snr_median": np.nan,
        "digitization_ks": np.nan,
        "digitization_asci": np.nan,
        "digitization_weighted_absolute_difference": np.nan,
    }

    print("challenge metrics test passed")


@func_indicator("testing trainer")
def test_trainer() -> None:
    """Test the trainer."""
    echo_write_permission(tmp_data_dir)
    echo_write_permission(tmp_model_dir)
    echo_write_permission(tmp_output_dir)

    train_config = deepcopy(TrainCfg)
    train_config.db_dir = tmp_data_dir
    # train_config.model_dir = model_folder
    # train_config.final_model_filename = "final_model.pth.tar"
    train_config.debug = True
    train_config.working_dir = tmp_model_dir / "working_dir"
    train_config.working_dir.mkdir(parents=True, exist_ok=True)

    train_config.n_epochs = 1
    train_config.batch_size = 4  # 16G (Tesla T4)
    # train_config.log_step = 20
    # # train_config.max_lr = 1.5e-3
    # train_config.early_stopping.patience = 20

    model_config = deepcopy(ModelCfg)
    model = MultiHead_CINC2024(config=model_config)
    if torch.cuda.device_count() > 1:
        model = DP(model)
        # model = DDP(model)
    model = model.to(device=DEVICE)
    if isinstance(model, DP):
        print("model size:", model.module.module_size, model.module.module_size_)
    else:
        print("model size:", model.module_size, model.module_size_)

    ds_train = CinC2024Dataset(train_config, training=True, lazy=True)
    ds_test = CinC2024Dataset(train_config, training=False, lazy=True)
    print(f"train size: {len(ds_train)}, test size: {len(ds_test)}")

    trainer = CINC2024Trainer(
        model=model,
        model_config=model_config,
        train_config=train_config,
        device=DEVICE,
        lazy=True,
    )
    # trainer._setup_dataloaders(ds_train, ds_test)
    # switch the dataloaders to make the test faster
    # the first dataloader is used for both training and evaluation
    # the second dataloader is used for validation only
    trainer._setup_dataloaders(ds_test, ds_train)

    best_model_state_dict = trainer.train()

    print(f"""Saved models: {list((Path(__file__).parent / "saved_models").iterdir())}""")

    print("trainer test passed")


@func_indicator("testing challenge entry")
def test_entry() -> None:
    """Test Challenge entry."""
    echo_write_permission(tmp_data_dir)
    echo_write_permission(tmp_model_dir)

    # run the model training function (script)
    print("run model training function")
    data_folder = tmp_data_dir
    train_digitization_model(str(data_folder), str(tmp_model_dir), verbose=2)
    train_dx_model(str(data_folder), str(tmp_model_dir), verbose=2)

    # run the model inference function (script)
    output_dir = tmp_output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print("run model for the original data")

    model_runner_args = CFG(
        data_folder=str(tmp_model_dir / SYNTHETIC_IMAGE_DIR),
        model_folder=str(tmp_model_dir),
        output_folder=str(output_dir),
        allow_failures=False,
        verbose=2,
    )
    model_runner_func(model_runner_args)

    print("evaluate model for the original data")

    model_evaluator_args = CFG(
        label_folder=str(tmp_model_dir / SYNTHETIC_IMAGE_DIR),
        output_folder=str(output_dir),
        extra_scores=True,
        score_file=None,
    )
    model_evaluator_func(model_evaluator_args)  # metrics are printed

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
        print("Please set CINC2024_REVENGER_TEST to true (1, y, yes, true, etc.) to run the test:")
        print("CINC2024_REVENGER_TEST=1 python test_docker.py")
        exit(0)

    print("#" * 80)
    print("testing team code")
    print("#" * 80)
    print(f"tmp_data_dir: {str(tmp_data_dir)}")
    print(f"tmp_model_dir: {str(tmp_model_dir)}")
    print(f"tmp_output_dir: {str(tmp_output_dir)}")
    print("#" * 80)

    test_dataset()  # passed
    # test_models()  # passed
    test_challenge_metrics()  # passed
    # test_trainer()  # directly run test_entry
    test_entry()
    # set_entry_test_flag(False)
