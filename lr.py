import numpy as np
import argparse
from typing import Tuple

VECTOR_LEN = 300 

def load_dataset(
    file: str
) -> Tuple[np.ndarray, np.ndarray]:
    dataset = np.loadtxt(file, delimiter='\t', comments=None, encoding='utf-8')
    label = np.array([row[0] for row in dataset])
    feature = np.array([row[1:] for row in dataset])
    return (feature, label)


def sigmoid(x : np.ndarray):
    """
    Implementation of the sigmoid function.

    Parameters:
        x (str): Input np.ndarray.

    Returns:
        An np.ndarray after applying the sigmoid function element-wise to the
        input.
    """
    e = np.exp(x)
    return e / (1 + e)


def dJ(
    theta : np.ndarray, 
    X : np.ndarray,  
    y : np.ndarray, 
    i : int
):  
    xi = np.concatenate(([1], X[i].T), axis=0)
    y = y.reshape(-1, 1)
    yi = y[i] 
    sigmoid_value = sigmoid(np.matmul(theta.T, xi))
    return ((sigmoid_value-yi) * xi).reshape(-1, 1) 

def J(
    theta : np.ndarray, 
    X : np.ndarray,  
    y : np.ndarray
): 
    j = 0
    for i in range(X.shape[0]):
        j += y[i]*np.log(sigmoid(theta.dot(X[i]))) + (1-y[i])*np.log(1-sigmoid(theta.T.dot(X[i])))
    return j*(-1)/X.shape[0]

def train(
    theta : np.ndarray, # shape (D,) where D is feature dim
    X : np.ndarray,     # shape (N, D) where N is num of examples
    y : np.ndarray,     # shape (N,)
    num_epoch : int, 
    learning_rate : float
) -> None:
    # TODO: Implement `train` using vectorization
    for epoch in range(num_epoch):
        for i in range(X.shape[0]):
            derivative = dJ(theta, X, y, i)  
            theta -= learning_rate * derivative
    return theta

def predict(
    theta : np.ndarray,
    X : np.ndarray,
    y: np.ndarray  
) -> np.ndarray:
    # TODO: Implement `predict` using vectorization
    y_pred = np.empty(shape=[X.shape[0], 1])
    for i in range(X.shape[0]):
        xi = np.concatenate(([1], X[i].T), axis=0)  
        y_pred[i] = sigmoid(np.dot(theta.T, xi))
    y_pred = np.asarray(y_pred >= 0.5).astype(int)
    count = 0
    y = y.reshape(-1,1)
    for i in range(y.shape[0]):
        if y_pred[i] != y[i]:
            print(y_pred[i], y[i])
            count+=1
    predict_error = count / y.shape[0]
    return y_pred, predict_error

def writetrain(file, train_predict):
    with open(file, 'w') as f:
        for i in range(train_predict.shape[0]-1):
            f.write(str(train_predict[i][0]) + '\n')
        f.write(str(train_predict[-1]))

def writetest(file, test_predict):
    with open(file, 'w') as f:
        for i in range(test_predict.shape[0]-1):
            f.write(str(test_predict[i][0]) + '\n')
        f.write(str(test_predict[-1]))

def writemetrics(file, train_error, test_error):
    with open(file, 'w') as f:
        f.write('error(train): ' + format(train_error, '.6f') + '\n')
        f.write('error(test): ' + format(test_error, '.6f'))

# def plot1(train_data, val_data, num_epoch, learning_rate, theta):
#     train_pred, train_error = predict(theta, X_train, y_train)
#     val_pred, val_error = predict(theta, X_val, y_val)
#     train_j = J(theta, X_train, train_pred)
    

if __name__ == '__main__':
    # This takes care of command line argument parsing for you!
    # To access a specific argument, simply access args.<argument name>.
    # For example, to get the learning rate, you can use `args.learning_rate`.
    parser = argparse.ArgumentParser()
    parser.add_argument("train_input", type=str, help='path to formatted training data')
    parser.add_argument("validation_input", type=str, help='path to formatted validation data')
    parser.add_argument("test_input", type=str, help='path to formatted test data')
    parser.add_argument("train_out", type=str, help='file to write train predictions to')
    parser.add_argument("test_out", type=str, help='file to write test predictions to')
    parser.add_argument("metrics_out", type=str, help='file to write metrics to')
    parser.add_argument("num_epoch", type=str, 
                        help='number of epochs of gradient descent to run')
    parser.add_argument("learning_rate", type=str, 
                        help='learning rate for gradient descent')
    args = parser.parse_args()
    
    X_train, y_train = load_dataset(args.train_input)
    X_test, y_test = load_dataset(args.test_input)
    X_val, y_val = load_dataset(args.validation_input)    

    theta = train(np.zeros(shape=[VECTOR_LEN+1, 1]), X_train,
                  y_train, int(args.num_epoch), float(args.learning_rate))
    train_pred, train_error = predict(theta, X_train, y_train)
    test_pred, test_error = predict(theta, X_test, y_test)

    writetrain(args.train_out, train_pred)
    writetest(args.test_out, test_pred)
    writemetrics(args.metrics_out, train_error, test_error)

    
