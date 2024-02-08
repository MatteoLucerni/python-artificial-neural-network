import matplotlib.pyplot as plt 
import numpy as np 
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, accuracy_score
import os
import struct

# fn to load MNIST dataset
def load_mnist(path="/"):
    train_labels_path = os.path.join(path,"train-labels.idx1-ubyte")
    train_images_path = os.path.join(path,"train-images.idx3-ubyte")
    
    test_labels_path = os.path.join(path,"t10k-labels.idx1-ubyte")
    test_images_path = os.path.join(path,"t10k-images.idx3-ubyte")
    
    labels_path = [train_labels_path, test_labels_path]
    images_path = [train_images_path, test_images_path]
        
    labels = []
    images = []
        
    for path in zip(labels_path, images_path):
        
        with open(path[0],'rb') as lbpath:
            magic, n = struct.unpack('>II', lbpath.read(8))
            lb = np.fromfile(lbpath, dtype=np.uint8)
            labels.append(lb)
            
        with open(path[1], 'rb') as imgpath:
            magic, num, rows, cols = struct.unpack(">IIII", imgpath.read(16))
            images.append(np.fromfile(imgpath, dtype=np.uint8).reshape(len(lb), 784))
            
    return images[0], images[1], labels[0], labels[1]

X_train, X_test, Y_train, Y_test = load_mnist(path="MNIST")

print("Proprietà: " + str(X_train.shape[1]))
print("Test: " + str(X_test.shape[0]))
print("Training: " + str(X_train.shape[0]))
print("Total pixels to guess: " +  str(X_train.shape[1] * X_test.shape[0]))

mms = MinMaxScaler()
X_train = mms.fit_transform(X_train)
X_test = mms.transform(X_test)

# test with logistic regression
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, Y_train)

Y_pred = lr.predict(X_test)
Y_proba = lr.predict_proba(X_test)

Y_pred_train = lr.predict(X_train)
Y_proba_train = lr.predict_proba(X_train)

acc = accuracy_score(Y_test, Y_pred)
acc_train = accuracy_score(Y_train, Y_pred_train)

lloss = log_loss(Y_test, Y_proba)
lloss_train = log_loss(Y_train, Y_proba_train)

print(f'TEST - Acc: {acc} / Loss: {lloss}')
print(f'TRAIN - Acc: {acc_train} / Loss: {lloss_train}')

print('=' * 100)

# Artificial Neural Network using Multilayers Perceptron (MLP)
from sklearn.neural_network import MLPClassifier

# One hidden layer with 100 nodes
mlp = MLPClassifier(hidden_layer_sizes=(100,), verbose=True)
mlp.fit(X_train, Y_train)

Y_pred = mlp.predict(X_test)
Y_proba = mlp.predict_proba(X_test)

Y_pred_train = mlp.predict(X_train)
Y_proba_train = mlp.predict_proba(X_train)

acc = accuracy_score(Y_test, Y_pred)
acc_train = accuracy_score(Y_train, Y_pred_train)

lloss = log_loss(Y_test, Y_proba)
lloss_train = log_loss(Y_train, Y_proba_train)

print(f'TEST - Acc: {acc} / Loss: {lloss}')
print(f'TRAIN - Acc: {acc_train} / Loss: {lloss_train}')

print('=' * 100)

# Two hidden layer with 512 nodes each
mlp = MLPClassifier(hidden_layer_sizes=(512, 512), verbose=True)
mlp.fit(X_train, Y_train)

Y_pred = mlp.predict(X_test)
Y_proba = mlp.predict_proba(X_test)

Y_pred_train = mlp.predict(X_train)
Y_proba_train = mlp.predict_proba(X_train)

acc = accuracy_score(Y_test, Y_pred)
acc_train = accuracy_score(Y_train, Y_pred_train)

lloss = log_loss(Y_test, Y_proba)
lloss_train = log_loss(Y_train, Y_proba_train)

print(f'TEST - Acc: {acc} / Loss: {lloss}')
print(f'TRAIN - Acc: {acc_train} / Loss: {lloss_train}')

print('=' * 100)

wrong_predictions = [i for i in range(len(Y_test)) if Y_test[i] != Y_pred[i]]
num_wrong_predictions = len(wrong_predictions)

# limit number of images
max_errors_to_show = min(25, num_wrong_predictions)

if num_wrong_predictions == 0:
    print("No wrong predictions")
else:
    print("Number of wrong predictions: " + str(num_wrong_predictions))
    cols = 5
    rows = int(np.ceil(max_errors_to_show / cols))

    plt.figure(figsize=(2*cols, 2*rows))

    for index in range(max_errors_to_show):
        wrong_index = wrong_predictions[index]
        image = X_test[wrong_index].reshape(28, 28)
        plt.subplot(rows, cols, index + 1)
        plt.imshow(image, cmap='gray')
        plt.title(f'True: {Y_test[wrong_index]}, Pred: {Y_pred[wrong_index]}')
        plt.axis('off')

    plt.tight_layout()
    plt.show()