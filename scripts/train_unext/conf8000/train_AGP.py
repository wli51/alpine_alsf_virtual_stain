# Unparallelized test script for training single UNeXt model on a single GPU.

from datetime import datetime
import gc
import pathlib
import sys

import torch
import mlflow
import pandas as pd

from virtual_stain_flow.transforms.normalizations import MaxScaleNormalize
from virtual_stain_flow.datasets.crop_cell_dataset import CropCellImageDataset
from virtual_stain_flow.datasets.cp_loaddata_dataset import CPLoadDataImageDataset
from virtual_stain_flow.datasets.aug_dataset import AugmentedBBoxImageDataset


# Load data and construct datasets

BATCH_NAME = "SN0313537"
INPUT_CHANNEL_NAMES = ["OrigBrightfield"]
TARGET_CHANNEL_NAMES = ["OrigAGP"]
CONFLUENCE = 8_000
EPOCHS = 300

DATASPLIT_OUTPUT_DIR = pathlib.Path(f"/projects/wli19@xsede.org/alsf_preprocess/{BATCH_NAME}/data_split_loaddata")
DATASPLIT_OUTPUT_DIR.resolve(strict=True)

if not DATASPLIT_OUTPUT_DIR.exists() and not DATASPLIT_OUTPUT_DIR.is_dir():
    print(f"Data split output directory {DATASPLIT_OUTPUT_DIR} does not exist.")
    sys.exit(1)

LOADDATA_FILE_PATH = DATASPLIT_OUTPUT_DIR / "loaddata_train.csv"
if not LOADDATA_FILE_PATH.exists() and not LOADDATA_FILE_PATH.is_file():
    print(f"LoadData file {LOADDATA_FILE_PATH} does not exist.")
    sys.exit(1)    

LOADDATA_HELDOUT_FILE_PATH = DATASPLIT_OUTPUT_DIR / "loaddata_heldout.csv"
if not LOADDATA_HELDOUT_FILE_PATH.exists() and not LOADDATA_HELDOUT_FILE_PATH.is_file():
    print(f"LoadData heldout file {LOADDATA_HELDOUT_FILE_PATH} does not exist.")
    sys.exit(1)

SC_FEATURES_DIR = pathlib.Path(
    f"/pl/active/koala/ALSF_pilot_data/preprocessed_profiles_{BATCH_NAME}/single_cell_profiles"
)
SC_FEATURES_DIR.resolve(strict=True)
if not SC_FEATURES_DIR.exists() and not SC_FEATURES_DIR.is_dir():
    print(f"Single-cell features directory {SC_FEATURES_DIR} does not exist.")
    sys.exit(1)
if not list(SC_FEATURES_DIR.glob("*.parquet")):
    print(f"No parquet files found in {SC_FEATURES_DIR}.")
    sys.exit(1)

TRAIN_ROOT = pathlib.Path('/scratch/alpine/wli19@xsede.org/')
TRAIN_ROOT.resolve(strict=False)
TRAIN_ROOT.mkdir(parents=True, exist_ok=True)
TRAIN_DIR = TRAIN_ROOT
TRAIN_DIR.mkdir(parents=True, exist_ok=True)
TRAIN_LOG_DIR = TRAIN_DIR / "mlruns"
TRAIN_LOG_DIR.mkdir(parents=True, exist_ok=True)
TRAIN_PLOT_DIR = TRAIN_DIR / "plots"
TRAIN_PLOT_DIR.mkdir(parents=True, exist_ok=True)
TMP_DIR = TRAIN_DIR / 'tmp'
TMP_DIR.mkdir(parents=True, exist_ok=True)

mlflow_track_uri = TRAIN_LOG_DIR.resolve().as_uri()
mlflow.set_tracking_uri(mlflow_track_uri)
print(f"MLflow tracking URI set to: {mlflow.get_tracking_uri()}")

experiment_name = "test_unext_experiment_scratch"
try:
    experiment_id = mlflow.create_experiment(
        name=experiment_name,
        tags={'purpose': 'test'}
    )
    print(f"Created MLflow experiment '{experiment_name}' with ID: {experiment_id}")
except Exception as e:
    if all(keyword in str(e).lower() for keyword in ['already', 'exists', 'experiment']):
        experiment = mlflow.get_experiment_by_name(experiment_name)
        experiment_id = experiment.experiment_id
        print(f"Experiment '{experiment_name}' already exists with ID: {experiment_id}")
    else:
        print(f"Experiment creation failed: {e}")
        sys.exit(1)

print(f"Filtering loaddata to only include seeding_density == {CONFLUENCE}.")

loaddata_df = pd.read_csv(LOADDATA_FILE_PATH)
print(f"Initial loaddata_df shape: {loaddata_df.shape}")
loaddata_df = loaddata_df.loc[loaddata_df['seeding_density'] == CONFLUENCE]
#loaddata_df = loaddata_df.sample(n=10, random_state=42)
print(f"Filtered loaddata_df shape: {loaddata_df.shape}")


loaddata_heldout_df = pd.read_csv(LOADDATA_HELDOUT_FILE_PATH)
print(f"Initial loaddata_heldout_df shape: {loaddata_heldout_df.shape}")
loaddata_heldout_df = loaddata_heldout_df.loc[loaddata_heldout_df['seeding_density'] == CONFLUENCE]
#loaddata_heldout_df = loaddata_heldout_df.sample(n=5, random_state=42)
print(f"Filtered loaddata_heldout_df shape: {loaddata_heldout_df.shape}")

sc_feature_files = list(
        SC_FEATURES_DIR.glob('*_sc_normalized.parquet')
    )

sc_features = pd.DataFrame()
for sc_features_parquet in sc_feature_files:
    if not sc_features_parquet.exists():
        print(f'{sc_features_parquet} does not exist, skipping...')
        continue 
    else:
        sc_features = pd.concat([
            sc_features, 
            pd.read_parquet(
                sc_features_parquet,
                columns=['Metadata_Plate', 'Metadata_Well', 'Metadata_Site', 'Metadata_Cells_Location_Center_X', 'Metadata_Cells_Location_Center_Y']
            )
        ])
print(f"Single-cell features shape: {sc_features.shape}")


cp_ids = CPLoadDataImageDataset(
    loaddata=loaddata_df,
    sc_feature=sc_feature_files,
    pil_image_mode='I;16',
)
crop_ds = CropCellImageDataset.from_dataset(
    cp_ids,
    patch_size=256,
    object_coord_x_field='Metadata_Cells_Location_Center_X',
    object_coord_y_field='Metadata_Cells_Location_Center_Y',
    fov=(1080, 1080)
)
# never forget to configure input and target channel keys for the crop
# dataset! Although the augmented dataset will be the one for training,
# the cropped dataset is still needed for visualization and plotting!
crop_ds.input_channel_keys = INPUT_CHANNEL_NAMES
crop_ds.target_channel_keys = TARGET_CHANNEL_NAMES
crop_ds.transform = MaxScaleNormalize(
    p=1, 
    normalization_factor=2**16 - 1,
)
aug_ds = AugmentedBBoxImageDataset.from_dataset(
    crop_ds,
    augment_to_n=10_000, #500,
)
aug_ds.input_channel_keys = INPUT_CHANNEL_NAMES
aug_ds.target_channel_keys = TARGET_CHANNEL_NAMES
aug_ds.transform = MaxScaleNormalize(
    p=1, 
    normalization_factor=2**16 - 1,
)

cp_ids_heldout = CPLoadDataImageDataset(
    loaddata=loaddata_heldout_df,
    sc_feature=sc_feature_files,
    pil_image_mode='I;16',
)
crop_ds_heldout = CropCellImageDataset.from_dataset(
    cp_ids_heldout,
    patch_size=256,
    object_coord_x_field='Metadata_Cells_Location_Center_X',
    object_coord_y_field='Metadata_Cells_Location_Center_Y',
    fov=(1080, 1080)
)
crop_ds_heldout.transform = MaxScaleNormalize(
    p=1, 
    normalization_factor=2**16 - 1,
)
crop_ds_heldout.input_channel_keys = INPUT_CHANNEL_NAMES
crop_ds_heldout.target_channel_keys = TARGET_CHANNEL_NAMES
batch = crop_ds_heldout[0]  # Test
if batch is None or not isinstance(batch, tuple):
    print("Heldout dataset sample retrieval failed.")
    sys.exit(1)
try:
    _i, _t = batch
    if _i is None or _t is None:
        print("Heldout dataset sample unpacking failed.")
        sys.exit(1)
    if not isinstance(_i, torch.Tensor) or not isinstance(_t, torch.Tensor):
        print("Heldout dataset sample unpacking returned non-tensor types.")
        sys.exit(1)
except Exception as e:
    print(f"Heldout dataset sample unpacking failed: {e}")
    sys.exit(1)

print(f"Training dataset length: {len(aug_ds)}")
print(f"Heldout dataset for visualization length: {len(crop_ds_heldout)}")

if not torch.cuda.is_available():
    print("CUDA is not available.")
    sys.exit(1)
n_cuda = torch.cuda.device_count()
if n_cuda < 1:
    print("No GPU available for training.")
    sys.exit(1)
print(f"Number of available GPUs: {n_cuda}")
print(f"Using GPU: {torch.cuda.get_device_name(0)}")
device = torch.device("cuda:0")


# Training configuration
print("Starting training configuration...")
run_name_prefix = f'UNeXt_density={CONFLUENCE}_hybrid_l1_msssim_loss_'
loss_weights = [
    1.0, # l1
    -1.0 # negative weight to MS-SSIM (higher=better)
]
tags = {
    'run_name': None,
    'architecture': 'ConvNeXtUNet',
    'depth': 4,
    'base_channels': 96,
    'down_sample_mode': '-',
    'up_sample_mode': 'pixelshuffle',
    'down_sample_block': '-',    
    'up_sample_block': 'convnext',
    'up_sample_block_count': 2,
    'output_activation': 'sigmoid',
    'input_channels': ['OrigBrightfield'],
    'target_channels': None,
    'dataset_length': len(crop_ds),
    'img_norm_method': 'maxscale',
    'generator_update_freq': 5,
    'discriminator_update_freq': 1,
    'optimizer': 'Adam',
    'optim_lr': 0.0002,
    'optim_betas': (0.5, 0.999),
    'discriminator_optimizer': 'Adam',
    'discriminator_optim_lr': 0.0002,
    'discriminator_optim_betas': (0.5, 0.999),
    'batch_size': 4,
    'loss': None,
    'loss_weights': loss_weights,
    'early_termination_metric': 'L1Loss',
}

from torch import nn
from torch import optim
from torchmetrics.image import MultiScaleStructuralSimilarityIndexMeasure
from albumentations import (
    HorizontalFlip, 
    VerticalFlip, 
    AdvancedBlur,
)

from virtual_stain_flow.metrics.PSNR import PSNR
from virtual_stain_flow.metrics.SSIM import SSIM

from virtual_stain_flow.metrics.MetricsWrapper import MetricsWrapper
from virtual_stain_flow.models.unext import ConvNeXtUNet
from virtual_stain_flow.trainers.logging_trainers.LoggingTrainer import LoggingTrainer
from virtual_stain_flow.vsf_logging.MlflowLogger import MlflowLogger
from virtual_stain_flow.vsf_logging.callbacks.PlotCallback import PlotPredictionCallback
from virtual_stain_flow.datasets.ram_cache_dataset import CompactRAMCache


loss = [
    nn.L1Loss(),
    MetricsWrapper(
        _metric_name='MS-SSIM',
        module=MultiScaleStructuralSimilarityIndexMeasure(
            data_range=1.0,
            kernel_size=11,
            sigma=1.5
        ).to(device)
    ),
]
tags['loss'] = loss
tags['target_channels'] = TARGET_CHANNEL_NAMES

print("Setting up cached dataset with augmentations...")
cached_dataset = CompactRAMCache(
    dataset=aug_ds,
    cache_size=None,
    
)
flip_transforms = [
    HorizontalFlip(p=0.5),
    VerticalFlip(p=0.5),
]    
cached_dataset.transform = flip_transforms
cached_dataset.input_only_transform = AdvancedBlur(
    p=0.25,
    blur_limit=(5, 5),
    sigma_x_limit=(0.1, 0.2),
    sigma_y_limit=(0.1, 0.2),
    beta_limit=(0.5, 1.5),
    noise_limit=(0.9, 1.1)
)

print("Initializing model and optimizer...")
run_name = run_name_prefix + str(TARGET_CHANNEL_NAMES)
tags['run_name'] = run_name

model = ConvNeXtUNet(
    in_channels=len(tags['input_channels']),
    out_channels=len(tags['target_channels']),
    decoder_up_block=tags['up_sample_mode'],
    decoder_compute_block=tags['up_sample_block'] ,
    act_type='sigmoid',
    _num_units=tags['up_sample_block_count'],
)

model_optim = optim.Adam(
    model.parameters(),
    lr=tags['optim_lr'],
    betas=tags['optim_betas']
)

_ = model.to(device)

print("Setting up logger and plotting callbacks...")
now = datetime.now()
timestamp_string = now.strftime("%Y-%m-%d-%H:%M:%S")

run_name = tags['run_name']
PLOT_DIR_TRAIN = TRAIN_PLOT_DIR /\
    ('plots_' + f'{run_name}_{timestamp_string}') 
PLOT_DIR_TRAIN.mkdir(exist_ok=True)

plot_callback = PlotPredictionCallback(
    name='plot_callback',
    save_path=PLOT_DIR_TRAIN,
    dataset=crop_ds,
    every_n_epochs=1,
    plot_metrics=[
        SSIM(_metric_name='ssim'), 
        PSNR(_metric_name='psnr')
        ],
    # kwargs passed to plotter
    show_plot=False,
    tag='plot_predictions_train'
)

PLOT_DIR_HELDOUT = TRAIN_PLOT_DIR /\
    ('plots_heldout_' + f'{run_name}_{timestamp_string}')
PLOT_DIR_HELDOUT.mkdir(exist_ok=True)

plot_callback_heldout = PlotPredictionCallback(
    name='plot_callback_heldout',
    save_path=PLOT_DIR_HELDOUT,
    dataset=crop_ds_heldout,
    every_n_epochs=1,
    plot_metrics=[
        SSIM(_metric_name='ssim'), 
        PSNR(_metric_name='psnr')
        ],
    # kwargs passed to plotter
    show_plot=False,
    tag='plot_predictions_heldout'
)

logger = MlflowLogger(
    name='logger',
    tracking_uri=str(TRAIN_LOG_DIR.resolve()),
    experiment_name=experiment_name,
    run_name=f'train_{run_name}',
    experiment_type='train',
    model_architecture='UNet+wGAN',
    target_channel_name=TARGET_CHANNEL_NAMES[0],
    tags={
        key: str(value) for key, value in tags.items()
    },
    mlflow_start_run_args={
        'nested': False
    },
    save_model_at_train_end=True,
    save_model_every_n_epochs=1,
    callbacks=[plot_callback, plot_callback_heldout]    
)

metric_fns = {
    "ssim_loss": SSIM(_metric_name="ssim"),
    "psnr_loss": PSNR(_metric_name="psnr"),
}

print("Starting training...")

trainer = LoggingTrainer(
    dataset=cached_dataset,

    model=model,
    optimizer=model_optim,
    backprop_loss=loss,
    backprop_loss_weights=loss_weights,

    patience=EPOCHS,
    train_for_epochs=EPOCHS,
    batch_size=tags['batch_size'],
    metrics=metric_fns,
    device=device,
    early_termination_metric=tags['early_termination_metric']
)

from time import time

start_time = time()
trainer.train(logger=logger)
logger.end_run()
end_time = time()
elapsed_time = end_time - start_time
print(f"Training time (seconds): {elapsed_time:.2f} for {EPOCHS} epochs.")

del model_optim
del model
del plot_callback_heldout
del plot_callback
del logger
del trainer
del cached_dataset
gc.collect()
print("Training complete.")