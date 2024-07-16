from math import e, log
from numpy import mean
import random

def forward_pass(w1, w2, b1, b2, x, y):
    z1 = x * w1 + b1
    a1 = max(0, z1)
    z2 = a1 * w2 + b2
    y_hat = 1 / (1 + e ** (-z2))
    
    return z1, a1, z2, y_hat

def compute_loss(y, y_hat):
    L = -y * log(y_hat) - (1 - y) * log(1 - y_hat)
    
    return L

def backward_pass(w2, x, y, y_hat, z1, a1):
    dy_hat = (-y / y_hat) + ((1 - y) / (1 - y_hat))
    dz2 = dy_hat * y_hat * (1 - y_hat)
    dw2 = dz2 * a1
    db2 = dz2
    
    da1 = dz2 * w2
    dz1 = da1 * (1 if z1 > 0 else 0)
    dw1 = dz1 * x
    db1 = dz1
    
    return dy_hat, dz2, dw2, db2, da1, dz1, dw1, db1
    
def update_weights(w1, w2, b1, b2, dw1, dw2, db1, db2, lr):
    w1 -= lr * dw1
    w2 -= lr * dw2
    b1 -= lr * db1
    b2 -= lr * db2
    
    return w1, w2, b1, b2

def train_model(initializations, lr, dataset, epochs):
    w1, w2, b1, b2 = initializations
    
    for epoch in range(epochs):
        losses = [0] * len(dataset)
        gradients = [(0, 0, 0, 0, 0, 0, 0, 0)] * len(dataset)
        
        for i in range(len(dataset)):
            z1, a1, z2, y_hat = forward_pass(w1, w2, b1, b2, dataset[i][0], dataset[i][1])
            
            losses[i] = compute_loss(dataset[i][1], y_hat)
            
            gradients[i] = backward_pass(w2, dataset[i][0], dataset[i][1], y_hat, z1, a1)
            
        L_avg = mean(losses)
        grads_avg = [0, 0, 0, 0, 0, 0, 0, 0]
        for i in range(len(gradients[0])):
            grads_avg[i] = mean([row[i] for row in gradients])
        dy_hat_avg, dz2_avg, dw2_avg, db2_avg, da1_avg, dz1_avg, dw1_avg, db1_avg = grads_avg
        
        print("epoch #" + str(epoch + 1))
        print("weight1: " + str(w1))
        print("weight2: " + str(w2))
        print("bias1: " + str(b1))
        print("bias2: " + str(b2))
        print("loss: " + str(L_avg))
        print()
        
        w1 = w1 - (lr * dw1_avg)
        w2 = w2 - (lr * dw2_avg)
        b1 = b1 - (lr * db1_avg)
        b2 = b2 - (lr * db2_avg)
        
weight1 = 1.1
weight2 = -0.5
bias1 = 0.6
bias2 = -0.2
learning_rate = 0.01
dataset = [
    (1, 0), (1, 0), (1, 0), (1, 0), (2, 0),
    (2, 1), (3, 0), (3, 0), (4, 0), (5, 1),
    (5, 0), (6, 1), (7, 1), (7, 0), (8, 1),
    (8, 0), (9, 1), (9, 1), (9, 1), (10, 1),
    (10, 1), (10, 1)
]
epochs = 10000

train_model([weight1, weight2, bias1, bias2], learning_rate, dataset, epochs)
