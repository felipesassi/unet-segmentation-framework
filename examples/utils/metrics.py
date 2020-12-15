import numpy as np

def dice_metric(y_true, y_predicted, thr=0.5):
        batch_size = y_true.shape[0]
        y_true = y_true.reshape(-1)
        y_predicted = y_predicted.reshape(-1)

        y_predicted[y_predicted < thr] = 0
        y_predicted[y_predicted >= thr] = 1 

        mult = 2*np.sum(y_true*y_predicted)
        sum_1 = np.sum(y_true)
        sum_2 = np.sum(y_predicted)
        dice = mult/(sum_1 + sum_2 + 1e-6)
        return np.mean(dice)

if __name__ == "__main__":
    pass