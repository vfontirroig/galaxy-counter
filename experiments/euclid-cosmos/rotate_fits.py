
import os
from astropy.io import fits
from astropy.wcs import WCS
from astropy.table import Table
from astropy.coordinates import SkyCoord
from astropy import units as u
import pandas as pd
from reproject import reproject_interp
from reproject.mosaicking import find_optimal_celestial_wcs
from astropy.visualization import ImageNormalize, PercentileInterval, AsinhStretch
import matplotlib.pyplot as plt

def rotate(fits_path):
    """
    Rotate a FITS image by the north aligned optimal angle.

    Args:
        fits_path: Path to the input FITS image

    Returns:
        hdu_aligned: In-memory PrimaryHDU holding the aligned data and updated WCS header,
            usable the same way as fits.open(...)[0] (hdu_aligned.data, hdu_aligned.header)
    """

    hdu_original = fits.open(fits_path)[0]

    # 1. Calculate the optimal North-aligned header for your data
    target_wcs, target_shape = find_optimal_celestial_wcs([hdu_original])

    # 2. Reproject and rotate the image data onto the new aligned WCS grid
    aligned_data, footprint = reproject_interp(hdu_original, target_wcs, shape_out=target_shape)

    # 3. Build an in-memory HDU instead of writing/reopening a FITS file on disk
    header_updated = target_wcs.to_header()
    hdu_aligned = fits.PrimaryHDU(data=aligned_data, header=header_updated)

    return hdu_aligned

def main():
    # Example usage
    
    print("Reading file")
    fits_file_ex = '/n03data/fontirro/cutouts/cosmos/256_cutouts/f115w/F115W_5.fits'


    hdu_aligned = rotate(fits_file_ex)
    wcs = WCS(hdu_aligned.header)
    print(wcs)
    print(hdu_aligned.data.shape)

    fig, ax = plt.subplots(1, 1, figsize=(5, 5), subplot_kw=dict(projection=wcs))
    ax.imshow(hdu_aligned.data, origin='lower', cmap='plasma', norm=ImageNormalize(hdu_aligned.data, interval=PercentileInterval(99.5), stretch=AsinhStretch()))

    out_path = '/n03data/fontirro/plots_examples/cosmos_rotation/aligned_image_example.png'
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')

    print("Saved figure.")

    # Optionally, save the aligned FITS to disk
    # output_path = "path/to/your/output_aligned.fits"  # Replace with desired output path
    # hdu_aligned.writeto(output_path, overwrite=True)
    # print(f"Aligned FITS saved to {output_path}")


if __name__ == "__main__":
    main()