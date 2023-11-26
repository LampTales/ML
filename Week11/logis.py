import numpy as np

class multi_class_logistic_regression(object):
    def __init__(self, X, y, learning_rate, max_epoch):
        self.X = X
        self.y = y
        self.learning_rate = learning_rate
        self.max_epoch = max_epoch
        self.w = np.zeros((X.shape[1], y.shape[1]))
        self.b = np.zeros(y.shape[1])

    def softmax(self, x):
        return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)
    
    def gradient(self):
        y_pred = self.softmax(np.dot(self.X, self.w) + self.b)
        dw = np.dot(self.X.T, y_pred - self.y) / self.X.shape[0]
        db = np.mean(y_pred - self.y, axis=0)
        return dw, db
    
    def train(self):
        for i in range(self.max_epoch):
            dw, db = self.gradient()
            self.w -= self.learning_rate * dw
            self.b -= self.learning_rate * db

    def get_weights(self):
        w = self.w.reshape(-1)
        b = self.b.reshape(-1)
        weights = np.concatenate((w, b))
        return weights



# input_file = open('0.in', 'r')


def read_data():
    # hyper_params = input_file.readline().split()
    hyper_params = input().split()
    N = int(hyper_params[0])
    D = int(hyper_params[1])
    C = int(hyper_params[2])
    E = int(hyper_params[3])
    L = float(hyper_params[4])
    X_train = []
    y_train = []
    for i in range(N):
        # line = input_file.readline().split()
        line = input().split()
        X_train.append([float(x) for x in line])
    for i in range(N):
        # line = input_file.readline().split()
        line = input().split()
        y_train.append([float(x) for x in line])
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    return N, D, C, E, L, X_train, y_train


def main():
    N, D, C, E, L, X_train, y_train = read_data()
    model = multi_class_logistic_regression(X_train, y_train, learning_rate=L, max_epoch=E)
    model.train()
    w = model.get_weights()
    for i in range(len(w)):
        print("%.3f" % w[i], end="\n")


if __name__ == '__main__':
    main()
