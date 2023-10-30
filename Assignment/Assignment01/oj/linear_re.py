import numpy as np

# read data from terminal
def read_data():
    line = input().split()
    N = int(line[0])
    M = int(line[1])
    X_train = []
    y_train = []
    for i in range(N):
        line = input().split()
        X_train.append(float(line[0]))
        y_train.append(float(line[1]))
    X_test = []
    for i in range(M):
        X_test.append(float(input()))
    return N, M, X_train, y_train, X_test


def transform(degree, X_train, X_test):
    X_train_transformed = []
    X_test_transformed = []
    for x in X_train:
        X_train_transformed.append([x ** i for i in range(degree + 1)])
    for x in X_test:
        X_test_transformed.append([x ** i for i in range(degree + 1)])
    return X_train_transformed, X_test_transformed


class LinearRegression:
    def __init__(self):
        self.w = None

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        self.w = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)

    def predict(self, X):
        X = np.array(X)
        return X.dot(self.w)


def main():
    N, M, X_train, y_train, X_test = read_data()
    X_train, X_test = transform(3, X_train, X_test)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pre = model.predict(X_test)
    for y in y_pre:
        print(y)


if __name__ == "__main__":
    main()
