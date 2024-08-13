"""
"""

import os
import shutil
from copy import deepcopy
from pathlib import Path
from typing import Union

import numpy as np
import torch
from torch.nn.parallel import DataParallel as DP
from torch.utils.data import DataLoader
from torch_ecg.cfg import CFG
from torch_ecg.utils.download import url_is_reachable
from torch_ecg.utils.misc import str2bool

from cfg import _BASE_DIR, ModelCfg, TrainCfg
from const import DATA_CACHE_DIR, MODEL_CACHE_DIR, REMOTE_MODELS
from data_reader import CINC2024Reader
from dataset import CinC2024Dataset, collate_fn
from evaluate_model import run as model_evaluator_func
from models import MultiHead_CINC2024
from outputs import CINC2024Outputs
from run_model import run as model_runner_func  # noqa: F401
from team_code import SubmissionCfg, train_models  # noqa: F401
from trainer import CINC2024Trainer
from utils.misc import func_indicator
from utils.scoring_metrics import compute_challenge_metrics, compute_classification_metrics, compute_digitization_metrics

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if ModelCfg.torch_dtype == torch.float64:
    torch.set_default_tensor_type(torch.DoubleTensor)
    DTYPE = np.float64
else:
    DTYPE = np.float32


os.environ["HUGGINGFACE_HUB_CACHE"] = str(MODEL_CACHE_DIR)
os.environ["HF_HUB_CACHE"] = str(MODEL_CACHE_DIR)
os.environ["HF_HOME"] = str(MODEL_CACHE_DIR.parent)


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
    assert len(ds_train) > 0, f"{len(ds_train) = }"
    assert len(ds_val) > 0, f"{len(ds_val) = }"

    # int indexing
    data = ds_val[0]
    assert isinstance(data, dict), f"{type(data) = }"
    assert "image" in data and "image_id" in data, f"{data.keys() = }"
    assert set(data.keys()) <= ds_val.data_fields, f"{set(data.keys()) = }, {ds_val.data_fields = }"
    # since the image backbone from transformers automatically converts the image to correct format
    # preprocessings including normalization, resizing, channel conversion, etc. are done in the backbone
    # instead of in the dataset
    assert isinstance(data["image"], np.ndarray), f"{type(data['image']) = }"
    assert data["image"].ndim == 3, f"{data['image'].shape = }"
    assert data["image"].shape[-1] == 3, f"{data['image'].shape = }"
    if "dx" in data:
        assert isinstance(data["dx"], np.ndarray) and data["dx"].ndim == 1
    if "digitization" in data:
        assert isinstance(data["digitization"], np.ndarray)
        assert data["digitization"].ndim == 2
        assert data["digitization"].shape[0] == ds_val.config.num_leads
    if "mask" in data:
        assert isinstance(data["mask"], np.ndarray)
        assert data["mask"].ndim == 2
        assert data["mask"].shape[0] == ds_val.config.num_leads
    if "bbox" in data:
        # TODO: test fields for object detection
        assert isinstance(data["bbox"], dict)
        assert set(data["bbox"].keys()) <= {"annotations", "image_id", "image_size", "format"}
        assert isinstance(data["bbox"]["annotations"], list)
        assert all([isinstance(ann, dict) for ann in data["bbox"]["annotations"]])
        assert all(
            [set(ann.keys()) <= {"bbox", "category_id", "is_crowd", "image_id", "area"} for ann in data["bbox"]["annotations"]]
        )

    # slice indexing
    batch_size = 4
    data = ds_val[:batch_size]
    assert isinstance(data, dict)
    assert "image" in data and "image_id" in data
    assert set(data.keys()) <= ds_val.data_fields
    assert isinstance(data["image"], list)
    assert len(data["image"]) == batch_size
    assert all([isinstance(img, np.ndarray) for img in data["image"]])
    if "dx" in data:
        assert isinstance(data["dx"], torch.Tensor)
        assert data["dx"].ndim == 2
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
    if "bbox" in data:
        assert isinstance(data["bbox"], list)
        assert len(data["bbox"]) == batch_size
        assert all([isinstance(bbox, dict) for bbox in data["bbox"]])

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
    ds_config.predict_bbox = False
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
        forward_output = model(model.get_input_tensors(input_images)["image"], labels=input_tensors)
        print(forward_output)
        if idx > 2:
            break

    # test classmethod "from_checkpoint"
    if url_is_reachable("https://drive.google.com/"):
        remote_model_source = "google-drive"
    else:
        remote_model_source = "deep-psp"

    if SubmissionCfg.digitizer is not None:
        pass

    if SubmissionCfg.detector is not None:
        pass

    if SubmissionCfg.classifier is not None:
        model, train_config = MultiHead_CINC2024.from_checkpoint(
            Path(MODEL_CACHE_DIR) / REMOTE_MODELS[SubmissionCfg.classifier]["filename"]
        )
        print("classifier loaded")
        print(model)
        print(train_config)
    for idx, input_tensors in enumerate(dl):
        print(model.inference(input_tensors["image"]))
        if idx > 2:
            break

    print("models test passed")


@func_indicator("testing challenge metrics")
def test_challenge_metrics() -> None:
    """Test the challenge metrics."""
    # test compute_classification_metrics
    labels = [{"dx": [["Acute MI", "AFIB/AFL"], [], ["Normal"]]}, {"dx": [["Normal"], ["Old MI", "PVC"]]}]
    outputs = [
        CINC2024Outputs(
            dx=[["Old MI", "AFIB/AFL"], ["HYP"], ["Normal"]],
            dx_classes=["NORM", "Acute MI", "Old MI", "STTC", "CD", "HYP", "PAC", "PVC", "AFIB/AFL", "TACHY", "BRADY"],
        ),
        CINC2024Outputs(
            dx=[[], ["PVC"]],
            dx_classes=["NORM", "Acute MI", "Old MI", "STTC", "CD", "HYP", "PAC", "PVC", "AFIB/AFL", "TACHY", "BRADY"],
        ),
    ]
    assert compute_classification_metrics(labels, outputs) == {"f_measure": 0.5333333333333333}

    # cm = ClassificationMetrics(multi_label=True)
    # cm(labels=np.array([0, 1, 1, 1, 1]), outputs=np.array([0, 1, 0, 0, 1]), num_classes=10)
    # assert cm.f1_measure == compute_classification_metrics(labels, outputs)["f_measure"]

    # test compute_digitization_metrics
    # TODO: implement the test, using labels and outputs containing digitization fields
    compute_digitization_metrics(labels, outputs)  # currently all nan

    # test compute_challenge_metrics
    assert compute_challenge_metrics(labels, outputs) == {
        "dx_f_measure": 0.5333333333333333,
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
    # train_config.db_dir = tmp_data_dir
    # train_config.model_dir = model_folder
    # train_config.final_model_filename = "final_model.pth.tar"
    train_config.db_dir = Path(DATA_CACHE_DIR)
    train_config.synthetic_images_dir = Path(DATA_CACHE_DIR) / "synthetic_images"
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
    # if torch.cuda.device_count() > 1:
    #     model = DP(model)
    #     # model = DDP(model)
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
    print("   Run model training function   ".center(80, "#"))
    data_folder = tmp_data_dir

    train_models(str(data_folder), str(tmp_model_dir), verbose=2)

    # run the model inference function (script)
    output_dir = tmp_output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print("   Run model   ".center(80, "#"))

    model_runner_args = CFG(
        # data_folder=str(tmp_model_dir / SYNTHETIC_IMAGE_DIR),
        data_folder=str(Path(DATA_CACHE_DIR) / "synthetic_images"),
        model_folder=str(tmp_model_dir),
        output_folder=str(output_dir),
        allow_failures=False,
        verbose=2,
    )
    model_runner_func(model_runner_args)

    print("   Evaluate model   ".center(80, "#"))

    # workaround for Dx prediction only:
    # copy the .dat files from the synthetic image folder to the output folder
    for src_file in (Path(DATA_CACHE_DIR) / "synthetic_images").rglob("*.dat"):
        dst_file = output_dir / src_file.relative_to(Path(DATA_CACHE_DIR) / "synthetic_images")
        if not dst_file.exists():
            dst_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_file, dst_file)
            print(f"copied {src_file} ---> {dst_file}")

    model_evaluator_args = CFG(
        folder_ref=str(Path(DATA_CACHE_DIR) / "synthetic_images"),
        folder_est=str(output_dir),
        extra_scores=True,
        score_file=None,
        no_shift=False,
    )
    model_evaluator_func(model_evaluator_args)  # metrics are printed

    print("Entry test passed")


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
    # test_challenge_metrics()  # passed
    # test_trainer()  # directly run test_entry
    test_entry()
    # set_entry_test_flag(False)
