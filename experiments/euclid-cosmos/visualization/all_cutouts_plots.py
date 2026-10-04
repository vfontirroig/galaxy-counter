import os
import pandas as pd
from astropy.io import fits
from astropy.wcs import WCS
from astropy.visualization import ImageNormalize, PercentileInterval, AsinhStretch
import matplotlib.pyplot as plt



OUT_DIR = '/n03data/fontirro/plots_examples/all_cutouts_plots'
CAT_FILE = '/n03data/fontirro/data_files/cat_crossmatch_v2.csv'

EUCLID_DIR_PATH = "/n03data/fontirro/cutouts/euclid/59_cutouts"  # base directory for Euclid cutouts.
COSMOS_DIR_PATH = "/n03data/fontirro/cutouts/cosmos/256_cutouts_new_rotated"  # base directory for COSMOS cutouts.


EUC_FIL = ['vis', 'nir_y', 'nir_j', 'nir_h']  # Euclid filters
COS_FIL = ['f115w', 'f150w', 'f277w']  # COSMOS filters

def crop(img, h, w):
    start_y = (img.shape[0] - h) // 2 
    start_x = (img.shape[1] - w) // 2 
    return img[start_y : start_y + h, start_x : start_x + w]

def main():
    # Load the catalog
    cat = pd.read_csv(CAT_FILE)

    # Create output directory if it doesn't exist
    os.makedirs(OUT_DIR, exist_ok=True)

    id_cos = 680102

    row = cat.loc[cat['id'] == id_cos]
    if row.empty:
        raise SystemExit(f"id {id_cos} not found in {CAT_FILE}")

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))

    # Every panel starts blank, so the ones with no cutout (the 4th COSMOS slot,
    # or any missing file) come out empty rather than as an empty labelled box.
    for ax in axes.flat:
        ax.axis('off')

    # Plot Euclid cutouts — top row
    for i, filt in enumerate(EUC_FIL):
        ax = axes[0, i]
        euclid_cutout_path = os.path.join(
            EUCLID_DIR_PATH, filt, row[f'59_raw_file_euclid_{filt}'].values[0]
        )
        if os.path.exists(euclid_cutout_path):
            with fits.open(euclid_cutout_path) as hdu:
                data = hdu[0].data
                ax.imshow(crop(data, 36, 36), origin='lower', cmap='plasma',
                          norm=ImageNormalize(data, interval=PercentileInterval(99.5),
                                              stretch=AsinhStretch()))
                ax.set_title(f"Euclid {filt.upper()}")
        else:
            print(f"File not found: {euclid_cutout_path}")


    # Plot COSMOS cutouts — bottom row
    tile = row['tile'].values[0]
    for i, filt in enumerate(COS_FIL):
        ax = axes[1, i]
        if filt == 'f115w':
            COSMOS_DIR_PATH = "/n03data/fontirro/cutouts/cosmos"
            cosmos_cutout_path = os.path.join(
                COSMOS_DIR_PATH, filt, f"{filt.upper()}_{id_cos}_{tile}.fits"
            )
            if os.path.exists(cosmos_cutout_path):
                with fits.open(cosmos_cutout_path) as hdu:
                    data = hdu[0].data
                    data = crop(data, 120, 120)
                    ax.imshow(data, origin='lower', cmap='plasma',
                            norm=ImageNormalize(data, interval=PercentileInterval(99.5),
                                                stretch=AsinhStretch()))
                    ax.set_title(f"COSMOS {filt.upper()}")
            else:
                print(f"File not found: {cosmos_cutout_path}")

        else:
            cosmos_cutout_path = os.path.join(
                COSMOS_DIR_PATH, filt, f"{filt.upper()}_{id_cos}_{tile}.fits"
            )
            if os.path.exists(cosmos_cutout_path):
                with fits.open(cosmos_cutout_path) as hdu:
                    data = hdu[0].data
                    data = crop(data, 120, 120)
                    ax.imshow(data, origin='lower', cmap='plasma',
                            norm=ImageNormalize(data, interval=PercentileInterval(99.5),
                                                stretch=AsinhStretch()))
                    ax.set_title(f"COSMOS {filt.upper()}")
            else:
                print(f"File not found: {cosmos_cutout_path}")


    fig.savefig(os.path.join(OUT_DIR, f"cutouts_{id_cos}.png"), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {os.path.join(OUT_DIR, f'cutouts_{id_cos}.png')}")



if __name__ == "__main__":
    main()