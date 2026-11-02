from tensorflow.keras.datasets import mnist
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np
import torch
from skimage.transform import resize
from skimage.filters import gaussian
from medmnist import PathMNIST
 
def load_mnist():
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    return X_train, y_train, X_test, y_test
    
def load_emnist():
    def fix_emnist_orientation_uint8(x_uint8):
        # x_uint8: torch uint8 tensor (N,28,28)
        return torch.rot90(x_uint8, 1, [1, 2]).flip(2)

    train_ds = datasets.EMNIST(root="data", split="letters", train=True, download=True)
    test_ds = datasets.EMNIST(root="data", split="letters", train=False, download=True)

    X_train_t = fix_emnist_orientation_uint8(train_ds.data)
    X_test_t = fix_emnist_orientation_uint8(test_ds.data)

    X_train = (X_train_t.numpy().astype(np.float32) / 255.0)
    X_test = (X_test_t.numpy().astype(np.float32) / 255.0)

    y_train = (train_ds.targets.numpy() - 1).astype(np.int64)
    y_test = (test_ds.targets.numpy() - 1).astype(np.int64)
    return X_train, y_train, X_test, y_test

def load_svhn(root="./data", split_train="train", transform=None, batch_size=512, num_workers=2, pin_memory=True):
    """
    Uses torchvision.datasets.SVHN (cropped 32x32) to produce:
      X_train: (N,32,32,3) uint8
      y_train: (N,) int64  labels 0..9 (SVHN uses 10 for '0' -> mapped to 0)
      X_test : (M,32,32,3) uint8
      y_test : (M,) int64
    """
    # Safe fallback: Ensure we always have a tensor transform so DataLoader won't crash
    if transform is None:
        transform = transforms.ToTensor()

    # Changed split from "extra" to "train" for standard SVHN pipeline
    train_set = datasets.SVHN(root=root, split=split_train, download=True, transform=transform)
    test_set  = datasets.SVHN(root=root, split="test",  download=True, transform=transform)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=pin_memory)
    test_loader  = DataLoader(test_set,  batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=pin_memory)

    def loader_to_numpy(dl):
        xs, ys = [], []
        for x, y in dl:
            # At this point, x is guaranteed to be a PyTorch Tensor
            if x.dtype.is_floating_point:
                x_u8 = (x.clamp(0, 1) * 255.0).to(torch.uint8)
            else:
                x_u8 = x.to(torch.uint8)
            
            x_u8 = x_u8.permute(0, 2, 3, 1).cpu().numpy()  # -> (B,32,32,3)
            y_np = y.cpu().numpy().astype(np.int64).reshape(-1)
            y_np[y_np == 10] = 0  # SVHN mapping convention

            xs.append(x_u8)
            ys.append(y_np)

        return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0)

    X_train, y_train = loader_to_numpy(train_loader)
    X_test,  y_test  = loader_to_numpy(test_loader)
    return X_train, y_train, X_test, y_test
    
def batch_rgb_to_grayscale(X_rgb):
    """
    Converts an array of shape (N, 32, 32, 3) uint8 RGB images
    into (N, 32, 32) uint8 Grayscale images using standard weights.
    """
    # Luma formula weights: 0.299*R + 0.587*G + 0.114*B
    weights = np.array([0.299, 0.587, 0.114], dtype=np.float32)
    
    # Dot product performs the weighted sum across the final axis (channels)
    # .astype(np.uint8) safely rounds the floating-point results back down
    return np.dot(X_rgb, weights).astype(np.uint8)
    
def svhn_normalize_for_hog(X, out_size=32):
    """
    Simpler, safer approach. Avoids segmenting mistakes while
    smoothing out SVHN high-frequency noise for cleaner HOG gradients.
    """
    X_out = []
    for img in np.arange(X.shape[0]):
        img_gray = X[img, :]
        x = img_gray.astype(np.float32)
        if x.max() > 1.5:
            x /= 255.0

        # 1. Light Gaussian blur to eliminate camera sensor noise / halftone dots
        # This prevents HOG from reacting to meaningless microscopic pixel variations
        x_smooth = gaussian(x, sigma=0.5, preserve_range=True).astype(np.float32)

        # 2. Directly resize to the target size 
        norm = resize(x_smooth, (out_size, out_size), order=1, anti_aliasing=True, preserve_range=True)
        
        # 3. Clip safely and return [0, 1] bounds. Let HOG's internal block-normalization do the rest.
        X_out.append(np.clip(norm, 0.0, 1.0))
    X_out = np.array(X_out)
    return X_out
    
def load_medmnist():
    train_dataset = PathMNIST(split="train")
    val_dataset = PathMNIST(split="val")
    test_dataset = PathMNIST(split="test")
    # 3. Extract the underlying NumPy arrays (.imgs for features, .labels for targets)
    X_train, y_train = train_dataset.imgs, train_dataset.labels
    X_val, y_val = val_dataset.imgs, val_dataset.labels
    X_test, y_test = test_dataset.imgs, test_dataset.labels

    # 4. Squeeze labels to flatten them from (N, 1) to (N,) 
    y_train = y_train.squeeze()
    y_val = y_val.squeeze()
    y_test = y_test.squeeze()
    return X_train,y_train, X_val, y_val, X_test, y_test
    
def stratified_train_val_split(X, y, val_frac=0.1, seed=0):
    X = np.asarray(X)
    y = np.asarray(y)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=val_frac, random_state=seed)
    (tr_idx, val_idx), = sss.split(X, y)
    return X[tr_idx], y[tr_idx], X[val_idx], y[val_idx]
