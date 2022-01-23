import numpy as np
from sklearn.model_selection import train_test_split

def avg_f1(precision, recall, num_classes):
    """

    :param precision: <list> list of precision scores for each class <list>
    :param recall: <list> list of recall scores for each class <list>
    :param num_classes: <int> number of classes <int>
    :return: <int> average f1 score of each class <int>
    """

    sum = 0

    for cls in num_classes:
        a = precision[cls] * recall[cls]
        b = precision[cls] + recall[cls]
        sum += (a / b)

    avg = 1/num_classes * sum
    return avg

def prod_f1(precision, recall, num_classes):
    """

    :param precision: <list> list of precision scores for each class <list>
    :param recall: <list> list of recall scores for each class <list>
    :param num_classes: <int> number of classes <int>
    :return: <int> product of f1 scores for each class <int>
    """

    sum = 0

    for cls in num_classes:
        a = precision[cls] * recall[cls]
        b = precision[cls] + recall[cls]
        sum *= (a / b)

    return sum

def prototype(X_c):
    """

    :param X_c: datapoints of class c
    :return: average features of class c
    """
    proto = np.zeros_like(X_c[0], dtype=np.float64)
    for instance in X_c:
        proto += instance

    proto = proto / X_c.shape[0]

    return proto