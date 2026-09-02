from tensorflow.keras.datasets import mnist
from torchvision import datasets, transforms
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np
 
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
    
def stratified_train_val_split(X, y, val_frac=0.1, seed=0):
    X = np.asarray(X)
    y = np.asarray(y)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=val_frac, random_state=seed)
    (tr_idx, val_idx), = sss.split(X, y)
    return X[tr_idx], y[tr_idx], X[val_idx], y[val_idx]
