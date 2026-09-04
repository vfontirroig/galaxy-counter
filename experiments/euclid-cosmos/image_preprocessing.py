'''
Script for preprocessing galaxy images (COSMOS and EUCLID).

Following the AION-1 Paper from Parker at al 2025.

'''
from __future__ import annotations
import torch


class CenterCrop:
    """Formatter that crops the images to have a fixed number of bands.
    i.e. It crops a square region of size crop_size × crop_size from the center of each image.
    """

    def __init__(self, crop_size: int = 96):
        self.crop_size = crop_size

    def __call__(self, image):
        _, _, height, width = image.shape
        start_x = (width - self.crop_size) // 2
        start_y = (height - self.crop_size) // 2
        return image[
            :, :, start_y : start_y + self.crop_size, start_x : start_x + self.crop_size
        ]


class Clamp:
    """Formatter that clamps the images to a given range."""

    def __init__(self):
        self.clamp_dict = BAND_CENTER_MAX

    def __call__(self, image, bands):
        for i, band in enumerate(bands):
            image[:, i, :, :] = torch.clip(
                image[:, i, :, :], -self.clamp_dict[band], self.clamp_dict[band]
            )
        return image
    

#COSMOS ZP per filter. This is obtained from the formula: ZP = -2.5*np.log10(hdr['PIXAR_SR']*[sr/pix] * 1e6) + 8.9  = 28.0865
#PIXAR_SR (for 30ms) is the pixel area in steradians. 1e6 converts to microJanskys. The constant 8.9 is a calibration offset.
#The formula was obtained from JWST documentation: https://jwst-docs.stsci.edu/jwst-near-infrared-camera/nircam-pipeline-reference/nircam-calibration-pipeline-reference/nircam-image-calibration/nircam-image-calibration-zeropoints
COSMOS_ZP = {
    "F150W": 28.1,
}


# Euclid AB zeropoints per filter — obtained from Euclid's fits file header.
EUCLID_ZP = {
    "VIS": 24.5,
    "Y":   24.3,   
    "J":   24.5,   
    "H":   24.4,      
}


# --- Common output basis: COSMOS-Web's native MJy/sr (surface brightness) ----
#
# The two surveys ship in different KINDS of unit, and neither is microjansky:
#   Euclid MER mosaics : BUNIT 'ADU/s' (VIS) / 'ELECTRON/s' (NISP) — a per-pixel
#       instrumental count rate. MER does not apply the flux calibration to the
#       pixels, it hands it over as the MAGZERO keyword (EUCLID_ZP above).
#   COSMOS-Web NIRCam  : BUNIT 'MJy/sr' — surface brightness, already calibrated
#       by the JWST pipeline's Stage 2 photom step.
#
# We convert onto MJy/sr rather than onto a per-pixel flux (uJy/px) basis.
# Surface brightness makes no reference to the pixel grid, so it divides out the
# ~11x solid-angle difference between Euclid's 0.1"/px and COSMOS-Web's 0.03"/px.
# That matters because RangeCompress's softening scale is an ABSOLUTE constant:
# on a per-pixel basis the 11x survives and lands the two surveys on different
# parts of the arcsinh curve (COSMOS compressed, Euclid effectively linear).

ARCSEC_IN_RAD = 4.8481368111e-6
# 1 MJy/sr expressed in uJy/arcsec^2  (1 MJy = 1e12 uJy)
UJY_PER_ARCSEC2_PER_MJY_SR = 1e12 * ARCSEC_IN_RAD ** 2   # 23.5044

EUCLID_PIXEL_SCALE_ARCSEC = 0.10  # MER mosaics; CD2_2 = 2.7778e-5 deg/px


def euclid_count_rate_to_mjy_sr(
    magzero: float,
    pixel_scale_arcsec: float = EUCLID_PIXEL_SCALE_ARCSEC,
) -> float:
    """Factor turning Euclid pixel values (ADU/s or e-/s) into MJy/sr.

    MAGZERO is quoted against the mosaic's own BUNIT, so it converts a pixel
    value into a per-pixel flux; dividing by the pixel area gives a surface
    brightness, which is then expressed in MJy/sr.
    """
    ujy_per_pixel = 10.0 ** ((23.9 - magzero) / 2.5)   # 23.9 == the ZP of microjansky
    ujy_per_arcsec2 = ujy_per_pixel / pixel_scale_arcsec ** 2
    return ujy_per_arcsec2 / UJY_PER_ARCSEC2_PER_MJY_SR


# band -> multiplicative factor onto MJy/sr. COSMOS-Web bands are already there
# and stay 1.0 for ANY of the released pixel scales (20/30/60 mas), precisely
# because surface brightness is grid-independent.
# NOTE: only VIS is verified against a real header (MAGZERO=24.5, DR1_R1). The
# NISP entries in EUCLID_ZP are unconfirmed — a NIR_Y tile reads MAGZERO=29.8
# with BUNIT 'ELECTRON/s', so check them before using Y/J/H.
BAND_TO_MJY_SR = {band: 1.0 for band in COSMOS_ZP}
BAND_TO_MJY_SR.update(
    {band: euclid_count_rate_to_mjy_sr(zp) for band, zp in EUCLID_ZP.items()}
)


class RescaleToCOSMOS:
    """Puts every band on COSMOS-Web's native basis: MJy/sr (surface brightness).

    COSMOS-Web bands pass through unchanged; Euclid bands are multiplied by
    BAND_TO_MJY_SR[band], derived from that mosaic's MAGZERO and pixel scale.

    Unknown bands raise rather than passing through. A missing entry used to mean
    "no rescaling needed", which is how COSMOS ended up ~47x too bright relative
    to Euclid: its pixels were treated as microjansky when they are MJy/sr.
    """

    def _scale(self, band: str) -> float:
        if band not in BAND_TO_MJY_SR:
            raise KeyError(
                f"No photometric calibration for band {band!r}. Add its Euclid "
                f"MAGZERO to EUCLID_ZP, or list it in COSMOS_ZP if it is already "
                f"in MJy/sr. Do not assume a band needs no rescaling."
            )
        return BAND_TO_MJY_SR[band]

    def forward(self, image: torch.Tensor, band: str) -> torch.Tensor:
        return image.clone() * self._scale(band)

    def backward(self, image: torch.Tensor, band: str) -> torch.Tensor:
        return image.clone() / self._scale(band)


class RangeCompress:
    """Formatter that applies arcsinh-based normalization. This formula comes from AION-1 Paper 
    (Parker et al 2025) and is used to compress the dynamic range of the images."""

    def __init__(self, range_compression_factor: float = 0.01, mult_factor: float = 10.0):
        """
        Initialize range compression.

        Args:
            range_compression_factor: Factor for arcsinh compression (default: 0.01)
            mult_factor: Multiplicative factor after compression (default: 10.0)
        """
        self.range_compression_factor = range_compression_factor
        self.mult_factor = mult_factor

    def forward(self, image):
        """
        Apply range compression: arcsinh(x / factor) * factor * mult_factor.
        Args:
            image: Input tensor

        Returns:
            Range-compressed tensor
        """
        image = image.clone()  # Avoid in-place modification
        image = (
            torch.arcsinh(image / self.range_compression_factor)
            * self.range_compression_factor
        )
        image = image * self.mult_factor
        return image

    def backward(self, image):
        """
        Reverse range compression.

        Args:
            image: Range-compressed tensor

        Returns:
            Decompressed tensor
        """
        image = image.clone()  # Avoid in-place modification
        image = image / self.mult_factor
        image = (
            torch.sinh(image / self.range_compression_factor)
            * self.range_compression_factor
        )
        return image


def get_survey(bands: list[str]) -> str:
    """
    Extract survey name from band names.

    Args:
        bands: List of band names (e.g., ['EUC-VIS', 'EUC-Y', ...])

    Returns:
        Survey name (e.g., 'EUC' or 'COS')
    """
    if not bands:
        raise ValueError("bands list cannot be empty")
    survey = bands[0].split("-")[0]
    return survey


def preprocess_image(
    image: torch.Tensor,
    bands: list[str],
    crop_size: int = 96,
    range_compression_factor: float = 0.01,
    mult_factor: float = 10.0,
    apply_range_compression: bool = True,
) -> torch.Tensor:
    """
    Apply full preprocessing pipeline to an image.

    Pipeline steps:
    1. Center crop to specified size
    2. Clamp values to band-specific ranges
    3. Rescale based on survey zeropoint
    4. (Optional) Apply range compression

    Args:
        image: Input image tensor with shape [batch, channels, height, width]
        bands: List of band names corresponding to channels
        crop_size: Size to crop to (default: 96)
        range_compression_factor: Factor for range compression (default: 0.01)
        mult_factor: Multiplicative factor for range compression (default: 10.0)
        apply_range_compression: Whether to apply range compression (default: True)

    Returns:
        Preprocessed image tensor
    """
    # Step 1: Center crop
    cropper = CenterCrop(crop_size=crop_size)
    processed = cropper(image)

    # Step 2: Clamp
    clamper = Clamp()
    processed = clamper(processed.clone(), bands)

    # Step 3: Rescale
    survey = get_survey(bands)
    rescaler = RescaleToCOSMOS()
    processed = rescaler.forward(processed.clone(), survey)

    # Step 4: Range compression (optional)
    if apply_range_compression:
        range_compressor = RangeCompress(
            range_compression_factor=range_compression_factor,
            mult_factor=mult_factor,
        )
        processed = range_compressor.forward(processed.clone())

    return processed



# Define ordered band lists for v2 lookup
EUC_BANDS = ["EUC-VIS", "EUC-Y", "EUC-J", "EUC-K"]
COSMOS_BANDS = ["COS-F115W", "COS-F150W", "COS-F277W", "COS-F444W"]


def preprocess_image_v2(
    image: torch.Tensor,
    crop_size: int = 120,
    survey: str = "cosmos",
    bands: list[str] | None = None,
) -> torch.Tensor:
    """
    Simplified preprocessing pipeline (V2).

    Infers bands from survey name ('cosmos' or 'euclid') unless `bands` is
    passed explicitly, in which case survey is ignored and any channel count works.
    Expects input shape (C, H, W) or (B, C, H, W).
    """
    # Handle dimensions: Ensure [Batch, Channel, H, W] for the classes
    is_batched = image.ndim == 4
    if not is_batched:
        if image.ndim == 3:
            image = image.unsqueeze(0) # (C, H, W) → (1, C, H, W)
        else:
            raise ValueError(f"Image must be 3D or 4D tensor, got shape {image.shape}")

    # Determine bands
    if bands is None:
        survey_key = survey.lower().strip()
        if survey_key == 'cos':
            bands = COSMOS_BANDS
        elif survey_key == 'euc':
            bands = EUC_BANDS
        else:
            raise ValueError(f"Unknown survey: '{survey}'. Supported: 'cosmos', 'euclid'")
        if image.shape[1] != len(bands):
            raise ValueError(
                f"Survey '{survey}' expects {len(bands)} channels, got {image.shape[1]}"
            )

    # Pipeline Execution

    # 1. Crop (Default 120)
    if crop_size is not None and crop_size > 0:
        cropper = CenterCrop(crop_size=crop_size)
        processed = cropper(image)
    else:
        processed = image.clone()

    # Clamp
    # clamper = Clamp()
    # processed = clamper(processed.clone(), bands)

    # 2. Rescale each band to COSMOS ZP. COSMOS cuouts are skipped.
    processed = processed.clone()
    rescaler = RescaleToCOSMOS()
    for i, band in enumerate(bands):
        processed[:, i, :, :] = rescaler.forward(processed[:, i:i+1, :, :], band)[:, 0, :, :]

    # 3. Range Compress (asinh).
    range_compression_factor = 0.01
    mult_factor = 10.0
    range_compressor = RangeCompress(
        range_compression_factor=range_compression_factor,
        mult_factor=mult_factor,
    )
    processed = range_compressor.forward(processed.clone())



    # Output handling
    # If input was not batched (3D), return 3D. If batched, return 4D.
    if not is_batched:
        processed = processed.squeeze(0) # (1, C, H, W) → (C, H, W)

    return processed


def main():
    """Demonstrate the preprocessing pipeline on one Euclid VIS and one COSMOS F150W cutout."""
    
    import numpy as np
    from astropy.io import fits
    import matplotlib.pyplot as plt
    import os
    from astropy.wcs import WCS
    from astropy.visualization import ImageNormalize, PercentileInterval, AsinhStretch

    #EUCLID_FILE = "/n03data/fontirro/euclid/40_cutouts/40_cutouts-vis/cutout_process_013_68b1674fTILE_101544256_14974135090968736_149.741351_2.147102_cutout.fits"
    COSMOS_FILE = "/n03data/fontirro/cutouts/cosmos/256_cutouts_rotated/f150w/F150W_5.fits"


    #label,filepath,hdu_index, band =  "EUC-VIS", EUCLID_FILE, 1, "VIS"
    label,filepath,hdu_index, band =  "COS-F150W", COSMOS_FILE, 0, "F150W"


    print("\n" + "=" * 60)
    print(f"PREPROCESSING PIPELINE — {label}")
    print("=" * 60)

    # Load image data.
    with fits.open(filepath) as hdul:
        data = hdul[hdu_index].data.astype(np.float32)
        wcs = WCS(hdul[hdu_index].header)
    if data.ndim == 2:
        data = data[np.newaxis]
    im_full = torch.from_numpy(data).unsqueeze(0)  # (1, 1, H, W)
    print(f"\n1. Original image shape: {im_full.shape}")
    print(f"   Range: [{im_full.min():.4f}, {im_full.max():.4f}]")

    # Step 1: Crop image to 120x120 (if needed)
    cropper = CenterCrop(crop_size=120)
    im_cropped = cropper(im_full)
    print(f"\n1. After cropper (crop_size=120): {im_cropped.shape}")
    print(f"   Range: [{im_cropped.min():.4f}, {im_cropped.max():.4f}]")

    #Checking cropped image
    # im_cropped_2d = im_cropped.squeeze().numpy()  # (1, 1, H, W) -> (H, W)

    # fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    # ax.imshow(im_cropped_2d, origin='lower', cmap='plasma', norm=ImageNormalize(im_cropped_2d, interval=PercentileInterval(99.5), stretch=AsinhStretch()))

    # out_path = '/n03data/fontirro/plots_examples/cosmos_rotation/aligned_image_example_preprocess.png'
    # os.makedirs(os.path.dirname(out_path), exist_ok=True)
    # plt.savefig(out_path, dpi=300, bbox_inches='tight')

    # print("Saved figure.")


    # Step 2: Rescale Euclid to COSMOS ZP (23.9); COSMOS passes through unchanged
    rescaler = RescaleToCOSMOS()
    im_rescaled = rescaler.forward(im_cropped.clone(), band)
    print(f"\n2. After rescale.forward (band={band}): {im_rescaled.shape}")
    print(f"   Range: [{im_rescaled.min():.4f}, {im_rescaled.max():.4f}]")


    # Step 3: Range compression (both COSMOS and EUCLID)
    range_compression_factor = 0.01
    mult_factor = 10.0
    range_compressor = RangeCompress(
        range_compression_factor=range_compression_factor,
        mult_factor=mult_factor,
    )
    im_range_compressed = range_compressor.forward(im_rescaled.clone())
    print(f"\n3. After range_compress: {im_range_compressed.shape}")
    print(f"   Range: [{im_range_compressed.min():.4f}, {im_range_compressed.max():.4f}]")
    print(f"   range_compression_factor: {range_compression_factor}")
    print(f"   mult_factor: {mult_factor}")
    print(f"   Formula: arcsinh(x / {range_compression_factor}) * {range_compression_factor} * {mult_factor}")
    

    im_cropped_2d = im_range_compressed.squeeze().numpy()  # (1, 1, H, W) -> (H, W)

    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.imshow(im_cropped_2d, origin='lower', cmap='plasma')

    out_path = '/n03data/fontirro/plots_examples/cosmos_rotation/aligned_image_example_preprocess.png'
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')

    print("Saved figure.")


    # # Summary
    # print("\n" + "=" * 60)
    # print("SUMMARY OF TRANSFORMATIONS")
    # print("=" * 60)
    # print(f"Original range:    [{im_full.min():.4f}, {im_full.max():.4f}]")
    # print(f"Rescaled range:    [{im_rescaled.min():.4f}, {im_rescaled.max():.4f}]")
    # if not is_euclid:
    #     print(f"Range compressed:  [{im_range_compressed.min():.4f}, {im_range_compressed.max():.4f}]")
    # print("=" * 60)

    # # Comparison: preprocess_image_v2 with explicit bands
    # print("\n" + "=" * 60)
    # print("USING preprocess_image_v2 FUNCTION: preprocess_image_v2()")
    # print("=" * 60)
    # im_preprocessed = preprocess_image_v2(im_full, bands=[band])
    # print(f"Preprocessed image shape: {im_preprocessed.shape}")
    # print(f"Preprocessed image range: [{im_preprocessed.min():.4f}, {im_preprocessed.max():.4f}]")
    # print("=" * 60)



if __name__ == "__main__":
    main()
