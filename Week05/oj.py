import numpy as np


def read_data():
    N = int(input())
    X_train = []
    y_train = []
    for i in range(N):
        line = input().split()
        y_train.append(int(line[0]))
        X_train.append([float(x) for x in line[1:]])

    M = int(input())
    X_test = []
    k_list = []
    for i in range(M):
        line = input().split()
        k_list.append(int(line[0]))
        X_test.append([float(x) for x in line[1:]])
    return N, M, X_train, y_train, k_list, X_test


class KNNClassifier:
    def __init__(self):
        self.X = None
        self.y = None

    def fit(self, X, y):
        self.X = np.array(X)
        self.y = np.array(y)

    def predict(self, X, K):
        X = np.array(X)
        y_pre = []
        for x, k in zip(X, K):
            distances = np.sqrt(np.sum((self.X - x) ** 2, axis=1))
            nearest = np.argsort(distances)[:k]
            y_pre.append(np.argmax(np.bincount(self.y[nearest])))
        return y_pre


def read_data_from_file():
    file = open('0.in', 'r')
    N = int(file.readline())
    X_train = []
    y_train = []
    for i in range(N):
        line = file.readline().split()
        y_train.append(int(line[0]))
        X_train.append([float(x) for x in line[1:]])
    M = int(file.readline())
    X_test = []
    k_list = []
    for i in range(M):
        line = file.readline().split()
        k_list.append(int(line[0]))
        X_test.append([float(x) for x in line[1:]])
    return N, M, X_train, y_train, k_list, X_test


def verify(y_pre):
    file = open('0.out', 'r')
    y_true = []
    for line in file.readlines():
        y_true.append(float(line))
    y_true = np.array(y_true)
    return np.sum(y_true == y_pre) / len(y_true)


def main():
    N, M, X_train, y_train, k_list, X_test = read_data()
    model = KNNClassifier()
    model.fit(X_train, y_train)
    y_pre = model.predict(X_test, k_list)

    for y in y_pre:
        print(y)
    # print(verify(y_pre))


if __name__ == "__main__":
    main()
