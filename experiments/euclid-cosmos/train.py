"""

Train the flow-matching model on paired Euclid (VIS) x COSMOS (F150W) cutouts.

Phase 1 (this script): simple pairs, no precomputed same-instrument neighbors.
  - encoder_1 conditions on the COSMOS counterpart of the same galaxy.
  - encoder_2 receives a random galaxy from the same instrument as the anchor,
    as a stand-in for real precomputed neighbors (see random_sameins below).
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
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint
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
    """Subclass that disables wandb-specific hooks from the base class.
    The base class assumes a wandb logger and 3-channel images; we use a CSV
    logger and 1-channel images, so those hooks are replaced with no-ops.
    """

    def __init__(self, *args, sample_dir=None, n_val_steps=50, **kwargs):
        super().__init__(*args, **kwargs)
        self.sample_dir = sample_dir
        self.n_val_steps = n_val_steps
        self._fixed_val_batch = None

    def on_train_start(self) -> None:
        import time
        self._train_start_time = time.time()
        print(f"\n{'='*60}")
        print(f"Training started - Target: {self.trainer.max_steps} steps")
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"{'='*60}\n")

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
            )
        return super().validation_step(batch, batch_idx)

    def on_validation_epoch_end(self) -> None:
        if self._fixed_val_batch is None or self.sample_dir is None:
            return

        anchor, cond, _, masks, galaxy_ids, surveys = (
            *[t.to(self.device) for t in self._fixed_val_batch[:4]],
            self._fixed_val_batch[4],
            self._fixed_val_batch[5],
        )
        os.makedirs(self.sample_dir, exist_ok=True)
        step = self.trainer.global_step

        # one plot per direction, containing only rows that match that direction
        direction_cfg = {
            "euclid": ("COSMOS to Euclid", "COSMOS input", "Generated Euclid", "Real Euclid", "Residual (Gen - Real)"),
            "cosmos": ("Euclid to COSMOS", "Euclid input", "Generated COSMOS", "Real COSMOS", "Residual (Gen - Real)"),
        }

        for survey, (dir_label, t0, t1, t2, t3) in direction_cfg.items():
            idx = [i for i, s in enumerate(surveys) if s == survey][:8]
            if not idx:
                continue

            anc = anchor[idx]
            con = cond[idx]
            msk = masks[idx]
            ids = [galaxy_ids[i] for i in idx]
            n = len(idx)

            sameins_vis = con.unsqueeze(1)
            with torch.no_grad():
                generated = self.sample(
                    cond_image_samegal=con,
                    cond_image_sameins=sameins_vis,
                    masks=msk,
                    num_steps=self.n_val_steps,
                )

            fig, axes = plt.subplots(n, 4, figsize=(7, 2.5 * n))
            if n == 1:
                axes = axes[None, :]
            for j, title in enumerate([t0, t1, t2, t3]):
                axes[0, j].set_title(title, fontsize=14)
            for i in range(n):
                for j, img in enumerate([con[i], generated[i], anc[i]]):
                    arr = img.squeeze().cpu().float().numpy()
                    axes[i, j].imshow(arr, cmap="plasma")
                    axes[i, j].axis("off")

                residual = (generated[i] - anc[i]).squeeze().cpu().float().numpy()
                vmax = np.abs(residual).max()
                axes[i, 3].imshow(residual, cmap="coolwarm", vmin=-vmax, vmax=vmax)
                axes[i, 3].axis("off")

                axes[i, 0].text(0.02, 0.98, f"idx={ids[i]}", fontsize=20,
                                ha="left", va="top", color="magenta",
                                transform=axes[i, 0].transAxes)

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
H5_PATH     = "/n03data/fontirro/data_files/euclid_cosmos_pairs_vis_f150w_v2.h5"
CKPT_DIR    = "/n03data/fontirro/euclid-cosmos/checkpoints/euclid-cosmos-vis-f150w/test-5-phase1/v2"  # where to save checkpoints and logs

BATCH_SIZE  = 64
NUM_WORKERS = 16
VAL_RATIO   = 0.05
TEST_RATIO  = 0.001
NUM_STEPS   = 250_000
IMAGE_SIZE  = 64      #Cutout spatial size
LR          = 1e-4    #learning rate for AdamW optimizer

N_GPUS      = 1       #set to number of GPUs on the node
# ---------------------------------------------------------------------------


def random_sameins(anchor: torch.Tensor, instrument: torch.Tensor) -> torch.Tensor:
    """For each row, randomly pick a different galaxy of the same instrument.

    anchor mixes instruments row-by-row (bidirectional dataset), so candidates
    are restricted to rows sharing the same `instrument` label — never the
    other instrument.

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

def collate_fn(batch):
    """
    arggs:
    batch: list of tuples (anchor, cond, metadata) from the dataset. Each anchor and cond have shape (1, H_SIZE, W_SIZE) and metadata is a dict with keys "idx" and "anchor_survey".

    Builds the 5-tuple the model expects:
      (anchor, samegal, sameins, masks, metadata)

    Direction (which survey is anchor vs condition) is determined by the
    dataset: even indices → Euclid anchor, odd indices → COSMOS anchor.
    """
    anchor = torch.stack([b[0] for b in batch])   # (B, 1, H, W)
    cond   = torch.stack([b[1] for b in batch])   # (B, 1, H, W) samegal counterpart (i.e the input)
    B = anchor.shape[0]
    metadata = [b[2] for b in batch]

    # 0 = euclid anchor, 1 = cosmos anchor — keeps random_sameins from ever crossing into the other instrument.
    instrument = torch.tensor([0 if m["anchor_survey"] == "euclid" else 1 for m in metadata])

    sameins = random_sameins(anchor, instrument).unsqueeze(1)  # (B, k=1, 1, H, W)
    masks    = torch.ones(B, 1, dtype=torch.bool)
    return anchor, cond, sameins, masks, metadata


def main():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    pl.seed_everything(42, workers=True)

    dataset   = EuclidCosmosDataset(H5_PATH, bidirectional=True) #returns ]euclid (anchor), cosmos (input), metadata or cosmos (anchor), euclid (input), metadata depending on the index.
    n_total   = len(dataset)
    n_test    = int(n_total * TEST_RATIO)
    n_val     = int(n_total * VAL_RATIO)
    n_train   = n_total - n_val - n_test
    generator = torch.Generator().manual_seed(42)
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
        collate_fn=collate_fn,
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
        collate_fn=collate_fn,
        persistent_workers=NUM_WORKERS > 0,
        pin_memory=True,
        prefetch_factor=4 if NUM_WORKERS > 0 else None,
    )

    model = EuclidCosmosModel(
        sample_dir=os.path.join(CKPT_DIR, "samples"),
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
        callbacks=[best_checkpoint, periodic_checkpoint],
        num_sanity_val_steps=2,
    )

    # Set to the latest checkpoint path to resume, or None to start fresh
    RESUME_FROM = None
    trainer.fit(model, train_loader, val_loader, ckpt_path=RESUME_FROM)


if __name__ == "__main__":
    main()
