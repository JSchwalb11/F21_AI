import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


def find_beta(x_1, y_1, x_2, z_1, size):
    numerator = z_1 - (x_1 * y_1)/size
    denom = x_2 - (x_1**2)/size
    return numerator/denom


def estimate_params(x,y):
    assert len(x) == len(y)
    x_1 = sum(x)
    y_1 = sum(y)
    x_2 = 0
    y_2 = 0
    z_1 = 0

    for i, element in enumerate(x):
        x_2 += x[i] * x[i]
        y_2 += y[i] * y[i]
        z_1 += x[i] * y[i]

    beta = find_beta(x_1, y_1, x_2, z_1, len(x))

    return beta

if __name__ == '__main__':
    X = [3,5,7,9,12,15,18]
    y = [100,250,330,590,660,780,890]
    test_x = [4,6,10,12]
    test_y = [3,5.5,6.5,9.0]
    month = np.asarray(30)

    beta = estimate_params(X,y)
    alpha = 0
    prediction = alpha + beta*month

    X = np.asarray(X)
    y = np.asarray(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=(0.1), random_state=1)
    X_train = X_train.reshape(-1, 1)
    y_train = y_train.reshape(-1, 1)
    X_test = X_test.reshape(-1, 1)
    month = month.reshape(-1, 1)
    regression_model = LinearRegression()
    regression_model.fit(X_train, y_train)
    coef = regression_model.coef_
    pred1 = alpha + beta*month
    pred2 = regression_model.predict(month)
    print("1a")
    print("Estimated beta: {0}".format(beta))
    print("Sklearn coef: {0}\n".format(regression_model.coef_[0][0]))

    print("1b")
    print("Estimated params prediction: {0}".format(pred1[0][0]))
    print("Sklearn regression prediction: {0}\n".format(pred2[0][0]))

    print("1c")
    if 1.5*890 < pred1:
        print("Use product XYZ")
    else:
        print("Use product ABC")



    # Plot outputs
    plt.scatter(X, y, color='black')
    plt.plot(X, beta*X + 0, 'r', label='Estimated Params')
    plt.plot(X, coef[0][0]*X + 0, 'b', label='Sklearn Regression')
    plt.scatter(month, beta * month + alpha, color='red')
    plt.scatter(month, coef * month + alpha, color='blue')
    plt.legend()

    plt.show()

