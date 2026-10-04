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



def main():
    # Load the catalog
    cat = pd.read_csv(CAT_FILE)

    # Create output directory if it doesn't exist
    os.makedirs(OUT_DIR, exist_ok=True)

    id_cos = 680102

    ax, fig = plt.subplots(2, 4, figsize=(16, 8))

    # Plot Euclid cutouts
    for i, filter in enumerate(EUC_FIL):
        euclid_cutout_path = os.path.join(EUCLID_DIR_PATH, filter, cat.loc[cat['id'] == id_cos]['59_raw_file_euclid_{filter}'].values[0])
        if os.path.exists(euclid_cutout_path):
            with fits.open(euclid_cutout_path) as hdu:
                data = hdu[0].data
                wcs = WCS(hdu[0].header)
                ax = ax[i]
                ax.imshow(data, origin='lower', cmap='plasma', norm=ImageNormalize(data, interval=PercentileInterval(99.5), stretch=AsinhStretch()))
                ax.set_title(f"Euclid {filter.upper()}")
                ax.axis('off')
        else:
            ax[i].text(0.5, 0.5, 'No Data', horizontalalignment='center', verticalalignment='center')
            ax[i].set_title(f"Euclid {filter.upper()}")
            ax[i].axis('off')

    plt.savefig(os.path.join(OUT_DIR, f"cutouts_{id_cos}.png"), dpi=300, bbox_inches='tight')


    # # Plot COSMOS cutouts
    # for i, filt in enumerate(COS_FIL):
    #     cosmos_cutout_path = os.path.join(COSMOS_DIR_PATH, f"{id}_{filt}.fits")
    #     if os.path.exists(cosmos_cutout_path):
    #         with fits.open(cosmos_cutout_path) as hdu:
    #             data = hdu[0].data
    #             wcs = WCS(hdu[0].header)
    #             ax = axes[i + len(EUC_FIL)]
    #             ax.imshow(data, origin='lower', cmap='plasma', norm=ImageNormalize(data, interval=PercentileInterval(99.5), stretch=AsinhStretch()))
    #             ax.set_title(f"COSMOS {filt.upper()}")
    #             ax.axis('off')
    #     else




if __name__ == "__main__":
    main()