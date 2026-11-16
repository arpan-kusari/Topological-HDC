import matplotlib.pyplot as plt
from topo_hdc.features.Image_processing import ProcessImage
from topo_hdc.data import load_emnist, stratified_train_val_split
from topo_hdc.corruptions import apply_cutout

if __name__ == "__main__":
    X_train, y_train, X_test, y_test = load_emnist()

    X_train, y_train, X_val, y_val = stratified_train_val_split(
        X_train,
        y_train,
        val_frac=0.2,
    )
    clean_image = X_train[0]
    corrupt_images = apply_cutout(X_test, size=12)
    corrupted_image = corrupt_images[0]
    # Use matched clean and corrupted images.
    clean_norm = ProcessImage(clean_image).normalize_glyph_gray(
        out_size=64,
        pad=2,
    )
    corrupt_norm = ProcessImage(corrupted_image).normalize_glyph_gray(
        out_size=64,
        pad=2,
    )

    fig, axes = plt.subplots(1, 4, figsize=(12, 3))

    for ax, image, title in zip(
        axes,
        [clean_image, corrupted_image, clean_norm, corrupt_norm],
        ["Clean input", "Cutout input", "Clean normalized", "Cutout normalized"],
    ):
        ax.imshow(image, cmap="gray", vmin=0)
        ax.set_title(title)
        ax.axis("off")

    plt.tight_layout()
    plt.show()
