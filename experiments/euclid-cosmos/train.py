"""

Train the flow-matching model on paired Euclid (VIS) x COSMOS (F150W) cutouts.

  - encoder_1 conditions on the COSMOS counterpart of the same galaxy.
  - encoder_2 receives the precomputed same-instrument neighbor of the anchor
    (sky-coordinate 1-NN: closest other galaxy in the same survey by angular
    separation, written into H5_PATH by experiments/euclid-cosmos/neighbors.py —
    run that script first). Rows with no neighbor found (neighbor_idx == -1)
    fall back to random_sameins so sameins is always populated.
  - lambda_geometric=0 matches the project default (see neighbours_train.py),
    independent of encoder_2/sameins: the geometric loss only compares
    encoder_1(target) against encoder_1(samegal counterpart).

Run locally (single GPU, for a quick sanity check):
    python experiments/euclid-cosmos/train.py

Submit on HPC via:
    sbatch experiments/euclid-cosmos/run_train.sh
"""

import os
import sys
import math
from functools import partial
import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Allow running from repo root or from the experiment directory
_here = os.path.dirname(__file__)
_repo_root = os.path.abspath(os.path.join(_here, "..", ".."))
sys.path.insert(0, _here)
sys.path.insert(0, os.path.join(_repo_root, "src"))
from dataset import EuclidCosmosDataset

from galaxy_counter.models.double_train_fm_neighbors import ConditionalFlowMatchingModule


class EuclidCosmosModel(ConditionalFlowMatchingModule):
    """
    Flow-matching model for Euclid VIS → COSMOS F150W and vice versa. 
    """

    def __init__(self, *args, sample_dir=None, n_val_steps=50,
                 input_plot_dir=None, input_plot_every_n_steps=500,
                 warmup_steps=1000, **kwargs):
        super().__init__(*args, **kwargs)
        self.sample_dir = sample_dir
        self.n_val_steps = n_val_steps
        self.input_plot_dir = input_plot_dir
        self.input_plot_every_n_steps = input_plot_every_n_steps
        self.warmup_steps = warmup_steps
        self._fixed_val_batch = None

    def configure_optimizers(self):
        """Linear warmup + cosine decay over the run's actual step budget.
        Overrides the base class's epoch-keyed CosineAnnealingLR, which is
        broken here: this run is driven by max_steps with max_epochs unset,
        so trainer.max_epochs resolves to -1, making T_max negative and the
        cosine schedule oscillate every epoch instead of decaying once."""
        optimizer = AdamW(self.parameters(), lr=self.lr)
        total_steps = self.trainer.max_steps
        warmup_steps = self.warmup_steps

        def lr_lambda(step):
            if step < warmup_steps:
                return step / max(1, warmup_steps)
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            progress = min(max(progress, 0.0), 1.0)
            return 0.5 * (1 + math.cos(math.pi * progress))

        scheduler = LambdaLR(optimizer, lr_lambda)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

    def on_train_start(self) -> None:
        import time
        self._train_start_time = time.time()
        print(f"\n{'='*60}")
        print(f"Training started - Target: {self.trainer.max_steps} steps")
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"{'='*60}\n")

    def training_step(self, batch, batch_idx):
        if (self.input_plot_dir is not None and self.trainer.is_global_zero
                and self.trainer.global_step % self.input_plot_every_n_steps == 0):
            self._plot_training_inputs(batch)
        return super().training_step(batch, batch_idx)

    def _plot_training_inputs(self, batch) -> None:
        """Diagnostic plot of the raw batch fed to the model at this step:
        the same-galaxy conditioning image, the same-instrument neighbor, and
        the anchor (generation target) — no model forward involved."""
        anchor, cond, sameins, _, metadata = batch
        n = min(8, anchor.shape[0])
        os.makedirs(self.input_plot_dir, exist_ok=True)
        step = self.trainer.global_step

        col_titles = ["Anchor (target)", "Samegal cond", "Sameins neighbor"]
        fig, axes = plt.subplots(n, 3, figsize=(7, 2.5 * n))
        if n == 1:
            axes = axes[None, :]
        for j, title in enumerate(col_titles):
            axes[0, j].set_title(title, fontsize=14)
        for i in range(n):
            imgs = [anchor[i], cond[i], sameins[i, 0]]
            for j, img in enumerate(imgs):
                arr = img.detach().squeeze().cpu().float().numpy()
                axes[i, j].imshow(arr, cmap="plasma")
                axes[i, j].axis("off")
            axes[i, 0].text(0.02, 0.98, f"idx={metadata[i]['idx']} ({metadata[i]['anchor_survey']})",
                            fontsize=10, ha="left", va="top", color="white",
                            transform=axes[i, 0].transAxes)
            sameins_idx = metadata[i]["sameins_idx"]
            sam_label = f"idx={sameins_idx}" if sameins_idx != -1 else "idx=? (random)"
            axes[i, 2].text(0.02, 0.98, sam_label,
                            fontsize=10, ha="left", va="top", color="white",
                            transform=axes[i, 2].transAxes)

        fig.suptitle(f"Training inputs  |  step {step}", fontsize=16, y=0.98)
        plt.tight_layout(rect=[0, 0, 1, 0.98])
        fname = os.path.join(self.input_plot_dir, f"train_inputs_step={step:07d}.png")
        plt.savefig(fname, dpi=100, bbox_inches="tight")
        plt.close()
        print(f"Saved training inputs: {fname}")

    def validation_step(self, batch, batch_idx):
        if self._fixed_val_batch is None and batch_idx == 0:
            anchor, cond, sameins, masks, metadata = batch
            n = anchor.shape[0]
            self._fixed_val_batch = (
                anchor[:n].detach().clone(),
                cond[:n].detach().clone(),
                sameins[:n].detach().clone(),
                masks[:n].detach().clone(),
                [m["idx"] for m in metadata[:n]],
                [m["anchor_survey"] for m in metadata[:n]],
                [m["sameins_idx"] for m in metadata[:n]],
            )
        return super().validation_step(batch, batch_idx)

    def on_validation_epoch_end(self) -> None:
        if self._fixed_val_batch is None or self.sample_dir is None:
            return

        anchor, cond, sameins, masks, galaxy_ids, surveys, sameins_ids = (
            *[t.to(self.device) for t in self._fixed_val_batch[:4]],
            self._fixed_val_batch[4],
            self._fixed_val_batch[5],
            self._fixed_val_batch[6],
        )
        os.makedirs(self.sample_dir, exist_ok=True)
        step = self.trainer.global_step
        num_samples = 3  # independent stochastic generations per row, same conditioning

        # one plot per direction, containing only rows that match that direction
        direction_cfg = {
            "euclid": ("COSMOS to Euclid", "COSMOS input", "Real Euclid"),
            "cosmos": ("Euclid to COSMOS", "Euclid input", "Real COSMOS"),
        }

        for survey, (dir_label, samegal_label, real_label) in direction_cfg.items():
            idx = [i for i, s in enumerate(surveys) if s == survey][:8]
            if not idx:
                continue

            anc = anchor[idx]
            con = cond[idx]
            sam = sameins[idx]
            msk = masks[idx]
            ids = [galaxy_ids[i] for i in idx]
            sam_ids = [sameins_ids[i] for i in idx]
            n = len(idx)

            with torch.no_grad():
                generations = [
                    self.sample(
                        cond_image_samegal=con,
                        cond_image_sameins=sam,
                        masks=msk,
                        num_steps=self.n_val_steps,
                    )
                    for _ in range(num_samples)
                ]
                # averaging the independent draws cancels per-sample stochastic
                # scatter, leaving what the model reproduces consistently
                mean_generation = torch.stack(generations).mean(dim=0)  # (n, 1, H, W)

            col_titles = (
                [samegal_label, "Same-instrument input", real_label]
                + [f"Generated #{k + 1}" for k in range(num_samples)]
                + [f"Mean of {num_samples} generations"]
            )
            n_cols = len(col_titles)

            fig, axes = plt.subplots(n, n_cols, figsize=(2.25 * n_cols, 2.5 * n))
            if n == 1:
                axes = axes[None, :]
            for j, title in enumerate(col_titles):
                axes[0, j].set_title(title, fontsize=12)
            for i in range(n):
                imgs = ([con[i], sam[i, 0], anc[i]]
                        + [gen[i] for gen in generations]
                        + [mean_generation[i]])
                for j, img in enumerate(imgs):
                    arr = img.squeeze().cpu().float().numpy()
                    axes[i, j].imshow(arr, cmap="plasma")
                    axes[i, j].axis("off")

                axes[i, 0].text(0.02, 0.98, f"idx={ids[i]}", fontsize=16,
                                ha="left", va="top", color="magenta",
                                transform=axes[i, 0].transAxes)

                sam_label = f"idx={sam_ids[i]}" if sam_ids[i] != -1 else "idx=? (random)"
                axes[i, 1].text(0.02, 0.98, sam_label, fontsize=16,
                                ha="left", va="top", color="magenta",
                                transform=axes[i, 1].transAxes)

            tag = dir_label.replace(" ", "").replace("to", "-")
            fig.suptitle(f"{dir_label}  |  step {step}", fontsize=18, y=0.98)
            plt.tight_layout(rect=[0, 0, 1, 0.97])
            fname = os.path.join(self.sample_dir, f"{tag}_step={step:07d}.png")
            plt.savefig(fname, dpi=100, bbox_inches="tight")
            plt.close()
            print(f"Saved samples: {fname}")

# ---------------------------------------------------------------------------
# CONFIG — edit before running
# ---------------------------------------------------------------------------
H5_PATH     = "/n03data/fontirro/data_files/euclid_cosmos_pairs_vis_f150w_v4_tilesB.h5"
CKPT_DIR    = "/n03data/fontirro/euclid-cosmos/checkpoints/euclid-cosmos-vis-f150w/test-7-phase1/v4"  # where to save checkpoints and logs

BATCH_SIZE  = 64
NUM_WORKERS = 16
VAL_RATIO   = 0.1
TEST_RATIO  = 0.05
NUM_STEPS   = 150_000
IMAGE_SIZE  = 64      #Cutout spatial size
LR          = 1e-4    #learning rate for AdamW optimizer

N_GPUS      = 1       #set to number of GPUs on the node
# ---------------------------------------------------------------------------


def random_sameins(anchor: torch.Tensor, instrument: torch.Tensor) -> torch.Tensor:
    """For each row, randomly pick a different galaxy of the same instrument.

    Args:
        anchor: (B, 1, H, W) tensor of galaxy images, mixed instruments.
        instrument: (B,) tensor of per-row instrument labels (e.g. 0/1).

    Returns:
        (B, 1, H, W) tensor — for each row i, a different row j with
        instrument[j] == instrument[i], excluding row i itself.
    """
    out = anchor.clone()
    for label in instrument.unique(): #overall: for each instrument, find the rows they share the same instrument and skip them. 
        group_idx = torch.nonzero(instrument == label).flatten() #gives a boolean mask. True if the instrument is the same as the galaxy's.
        n = group_idx.numel() #how many rows belong to the instrument
        if n <= 1:
            continue
        local = torch.randint(0, n - 1, (n,), device=anchor.device) #devide is used so it can run with a cpu or gpu without any problem. 
        local = local + (local >= torch.arange(n, device=anchor.device))  # skip self
        out[group_idx] = anchor[group_idx[local]]
    return out  # (B, 1, H, W)

def collate_fn(batch, dataset):
    """
    Args:
        batch: list of tuples (anchor, cond, metadata) from the dataset. Each anchor and cond have
            shape (1, H_SIZE, W_SIZE) and metadata is a dict with keys "idx" and "anchor_survey".
        dataset: the EuclidCosmosDataset instance, used to fetch the precomputed same-instrument
            neighbor image (neighbor_idx_sky_euclid / neighbor_idx_sky_cosmos, written by
            experiments/euclid-cosmos/neighbors.py) directly from its already-open HDF5 handle.

    Builds the 5-tuple the model expects:
      (anchor, samegal, sameins, masks, metadata)

    sameins is the precomputed same-instrument neighbor (sky-coordinate 1-NN:
    closest other galaxy in the same survey by true angular separation, which
    is what actually shares observing conditions/PSF — see nearest_neighbor_sky
    in neighbors.py). Rows with no precomputed neighbor (neighbor_idx == -1,
    e.g. beyond MAX_NEIGHBOR_SEP_ARCSEC) fall back to random_sameins so
    sameins is always populated and masks stays all-True, same as before.

    Direction (which survey is anchor vs condition) is determined by the
    dataset: even indices → Euclid anchor, odd indices → COSMOS anchor.
    """
    anchor = torch.stack([b[0] for b in batch])   # (B, 1, H, W)
    cond   = torch.stack([b[1] for b in batch])   # (B, 1, H, W) samegal counterpart (i.e the input)
    metadata = [b[2] for b in batch]
    B = anchor.shape[0]

    dataset._open_file()
    f = dataset.file

    sameins = anchor.clone().unsqueeze(1)  # (B, k=1, 1, H, W); overwritten below when a precomputed neighbor exists
    missing = []
    for i, meta in enumerate(metadata):
        idx, survey = meta["idx"], meta["anchor_survey"]
        if survey == "euclid":
            neighbor_key, img_key, norm_key = "neighbor_idx_sky_euclid", "euclid_images_upscaled", "euclid_up"
        else:
            neighbor_key, img_key, norm_key = "neighbor_idx_sky_cosmos", "cosmos_images_downscaled", "cosmos_ds"

        j = int(f[neighbor_key][idx, 0])
        meta["sameins_idx"] = j  # -1 until random_sameins fills it in below, if this row is missing
        if j == -1:
            missing.append(i)
            continue

        raw = torch.from_numpy(f[img_key][j].copy())
        mean, std = dataset.norm_dict[norm_key]
        sameins[i, 0] = (raw - mean) / std

    if missing:
        # random_sameins doesn't report which row it picked, so these stay
        # sameins_idx == -1 (shown as "random" rather than a real row index).
        instrument = torch.tensor([0 if m["anchor_survey"] == "euclid" else 1 for m in metadata])
        fallback = random_sameins(anchor, instrument)
        for i in missing:
            sameins[i, 0] = fallback[i]

    masks = torch.ones(B, 1, dtype=torch.bool)  # sameins is always populated (precomputed or fallback)
    return anchor, cond, sameins, masks, metadata


def main():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    pl.seed_everything(42, workers=True)

    dataset   = EuclidCosmosDataset(H5_PATH, bidirectional=True) #returns euclid (anchor), cosmos (input), metadata or cosmos (anchor), euclid (input), metadata depending on the index.
    n_total   = len(dataset)
    n_test    = int(n_total * TEST_RATIO)
    n_val     = int(n_total * VAL_RATIO)
    n_train   = n_total - n_val - n_test
    generator = torch.Generator().manual_seed(123)
    train_ds, val_ds, test_ds = random_split(dataset, [n_train, n_val, n_test], generator=generator)

    print(f"Dataset: {n_total} pairs → {n_train} train / {n_val} val / {n_test} test")

    # Save split indices so evaluation scripts use the exact same sets
    os.makedirs(CKPT_DIR, exist_ok=True)
    np.save(os.path.join(CKPT_DIR, "test_indices.npy"), np.array(test_ds.indices))
    np.save(os.path.join(CKPT_DIR, "val_indices.npy"),  np.array(val_ds.indices))
    print(f"Test indices saved to: {os.path.join(CKPT_DIR, 'test_indices.npy')}")
    print(f"Val  indices saved to: {os.path.join(CKPT_DIR, 'val_indices.npy')}")

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        collate_fn=partial(collate_fn, dataset=dataset),
        persistent_workers=NUM_WORKERS > 0,
        pin_memory=True,
        prefetch_factor=4 if NUM_WORKERS > 0 else None,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=partial(collate_fn, dataset=dataset),
        persistent_workers=NUM_WORKERS > 0,
        pin_memory=True,
        prefetch_factor=4 if NUM_WORKERS > 0 else None,
    )

    model = EuclidCosmosModel(
        sample_dir=os.path.join(CKPT_DIR, "samples"),
        input_plot_dir=os.path.join(CKPT_DIR, "train_inputs"), #added for the training inputs diagnostic plot.
        input_plot_every_n_steps=1000, #same as above.
        in_channels=1,            # Euclid VIS: 1 channel
        cond_channels=1,          # COSMOS F150W: 1 channel
        image_size=IMAGE_SIZE,
        model_channels=128,
        channel_mult=(1, 2, 4, 4),
        cross_attention_dim=16,
        pretrained_encoder=False,
        concat_conditioning=False,
        lr=LR,
        num_sample_images=8,
        num_mse_images=32,
        num_integration_steps=250,
        lambda_generative=1.0,
        lambda_geometric=0.0,     # no neighbors yet → geometric loss disabled
        mask_center=False,
    )

    csv_logger = CSVLogger(save_dir=CKPT_DIR, name="logs")

    os.makedirs(CKPT_DIR, exist_ok=True)
    best_checkpoint = ModelCheckpoint(
        dirpath=CKPT_DIR,
        monitor="val/loss",
        mode="min",
        save_top_k=1,
        filename="best-epoch={epoch:02d}-step={step}",
        auto_insert_metric_name=False,
    )
    periodic_checkpoint = ModelCheckpoint(
        dirpath=CKPT_DIR,
        every_n_train_steps=2000,
        save_top_k=1,
        filename="latest-step={step}",
    )
    early_stopping = EarlyStopping(
        monitor="val/loss",
        mode="min",
        patience=20,  # in validation checks, i.e. 20 * val_check_interval = 20_000 steps
        verbose=True,
    )

    trainer = pl.Trainer(
        max_steps=max(1, int(NUM_STEPS / N_GPUS)),
        logger=csv_logger,
        accelerator="auto",
        devices=N_GPUS,
        strategy="ddp_find_unused_parameters_true" if N_GPUS > 1 else "auto",
        log_every_n_steps=10,
        precision="bf16-mixed",
        val_check_interval=1000,
        check_val_every_n_epoch=None,
        callbacks=[best_checkpoint, periodic_checkpoint, early_stopping],
        num_sanity_val_steps=2,
    )

    # Set to the latest checkpoint path to resume, or None to start fresh
    RESUME_FROM = None
    trainer.fit(model, train_loader, val_loader, ckpt_path=RESUME_FROM)


if __name__ == "__main__":
    main()
