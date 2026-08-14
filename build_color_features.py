import os
import numpy as np
from PIL import Image


IMAGE_DIR = "data/images"
FILES_FILE = "data/valid_files.npy"

OUTPUT_FILE = "data/color_features.npy"


# --------------------------------
# Settings
# --------------------------------

H_BINS = 18
S_BINS = 8


# --------------------------------
# RGB -> HSV
# --------------------------------

def rgb_to_hsv(image):
    """
    Convert RGB image array from [0,255]
    to HSV values in [0,1].
    """

    rgb = image.astype(np.float32) / 255.0

    r = rgb[:, :, 0]
    g = rgb[:, :, 1]
    b = rgb[:, :, 2]

    max_val = np.max(rgb, axis=2)
    min_val = np.min(rgb, axis=2)

    delta = max_val - min_val

    h = np.zeros_like(max_val)

    mask = delta != 0

    # Red is max
    mask_r = mask & (max_val == r)

    h[mask_r] = (
        ((g[mask_r] - b[mask_r]) / delta[mask_r])
        % 6
    )

    # Green is max
    mask_g = mask & (max_val == g)

    h[mask_g] = (
        (b[mask_g] - r[mask_g]) / delta[mask_g]
    ) + 2

    # Blue is max
    mask_b = mask & (max_val == b)

    h[mask_b] = (
        (r[mask_b] - g[mask_b]) / delta[mask_b]
    ) + 4

    h /= 6.0

    s = np.zeros_like(max_val)

    nonzero_max = max_val != 0

    s[nonzero_max] = (
        delta[nonzero_max]
        / max_val[nonzero_max]
    )

    v = max_val

    return h, s, v


# --------------------------------
# Build colour feature
# --------------------------------

def extract_color_feature(image_path):

    image = Image.open(
        image_path
    ).convert("RGB")

    # Resize to reduce computation.
    image = image.resize(
        (270, 360)
    )

    image_array = np.array(
        image
    )

    h, s, v = rgb_to_hsv(
        image_array
    )

    # Ignore mostly gray/white background.
    #
    # Saree colours generally have more
    # saturation than the background.
    mask = (
        (s > 0.15)
        &
        (v > 0.15)
    )

    h_values = h[mask]
    s_values = s[mask]

    if len(h_values) == 0:

        return np.zeros(
            H_BINS * S_BINS,
            dtype=np.float32
        )

    histogram, _, _ = np.histogram2d(
        h_values,
        s_values,
        bins=[
            H_BINS,
            S_BINS
        ],
        range=[
            [0.0, 1.0],
            [0.0, 1.0]
        ]
    )

    # Normalize
    histogram = histogram.flatten()

    norm = np.linalg.norm(
        histogram
    )

    if norm > 0:

        histogram /= norm

    return histogram.astype(
        np.float32
    )


# --------------------------------
# Main
# --------------------------------

print("Loading filenames...")

valid_files = np.load(
    FILES_FILE,
    allow_pickle=True
)

print(
    f"Images: {len(valid_files)}"
)

features = []

failed = 0


for i, filename in enumerate(
    valid_files
):

    filename = str(filename)

    image_path = os.path.join(
        IMAGE_DIR,
        filename
    )

    try:

        feature = extract_color_feature(
            image_path
        )

        features.append(feature)

    except Exception as e:

        print(
            f"[FAILED] {filename}"
        )

        print(e)

        features.append(
            np.zeros(
                H_BINS * S_BINS,
                dtype=np.float32
            )
        )

        failed += 1

    if (
        (i + 1) % 100 == 0
        or i == len(valid_files) - 1
    ):

        print(
            f"Processed "
            f"{i + 1}/{len(valid_files)}"
        )


features = np.stack(
    features
)


np.save(
    OUTPUT_FILE,
    features
)


print("\n================================")
print("COLOUR FEATURES COMPLETE")
print("================================")

print(
    "Feature shape:",
    features.shape
)

print(
    "Failed:",
    failed
)

print(
    "Saved:",
    OUTPUT_FILE
)