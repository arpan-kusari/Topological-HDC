 import numpy as np
 
 def accuracy(y_true, y_pred):
    return (y_true == y_pred).mean()
