# importazione pacchetti necessari
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import log_loss, accuracy_score
import os
import struct

#! =======================================================================
#! SUDDIVISIONE IN PIU' SUB SETS (TRAIN e TEST)
#! =======================================================================


# funzione per caricare il dataset MNIST
def load_mnist(path="/"):
    train_labels_path = os.path.join(path, "train-labels.idx1-ubyte")
    train_images_path = os.path.join(path, "train-images.idx3-ubyte")

    test_labels_path = os.path.join(path, "t10k-labels.idx1-ubyte")
    test_images_path = os.path.join(path, "t10k-images.idx3-ubyte")

    labels_path = [train_labels_path, test_labels_path]
    images_path = [train_images_path, test_images_path]

    labels = []
    images = []

    for path in zip(labels_path, images_path):

        with open(path[0], "rb") as lbpath:
            magic, n = struct.unpack(">II", lbpath.read(8))
            lb = np.fromfile(lbpath, dtype=np.uint8)
            labels.append(lb)

        with open(path[1], "rb") as imgpath:
            magic, num, rows, cols = struct.unpack(">IIII", imgpath.read(16))
            images.append(np.fromfile(imgpath, dtype=np.uint8).reshape(len(lb), 784))

    return images[0], images[1], labels[0], labels[1]


# divisione del dataset in subsets di TRAIN e di TEST
X_train, X_test, Y_train, Y_test = load_mnist(path="MNIST")

print("Proprietà: " + str(X_train.shape[1]))
print("Test: " + str(X_test.shape[0]))
print("Training: " + str(X_train.shape[0]))
print("Total pixels to guess: " + str(X_train.shape[1] * X_test.shape[0]))

#! =======================================================================
#! PRE ELABORAZIONE DEI DATI
#! =======================================================================

# standardizzazione dei dati
mms = MinMaxScaler()
X_train = mms.fit_transform(X_train)
X_test = mms.transform(X_test)

#! =======================================================================
#! SCEGLIERE I PARAMETRI INZIALI DELL'ALGORITMO
#! =======================================================================

from sklearn.neural_network import MLPClassifier

# addestro la rete neurale sui dati di TRAIN
mlp = MLPClassifier(
    hidden_layer_sizes=(100, 30), verbose=True
)  # ? Un singolo Hidden Layer con 100 nodi

#! =======================================================================
#! USARE I DATI DI TRAIN PER FAR IMPARARE L'ALGORITMO
#! =======================================================================

mlp.fit(X_train, Y_train)

#! =======================================================================
#! USARE I DATI DI TEST PER METTERLO ALLA PROVA
#! =======================================================================

# eseguio le predizioni sui dati di TEST
Y_pred = mlp.predict(X_test)
Y_proba = mlp.predict_proba(X_test)

# eseguo le predizioni sui dati di TRAIN per verificare overfitting
Y_pred_train = mlp.predict(X_train)
Y_proba_train = mlp.predict_proba(X_train)

#! =======================================================================
#! UTILIZZARE METRICHE PER VALUTARE COME HA PERFORMATO
#! =======================================================================

# calcolo le metriche di Accuracy e di Loss
acc = accuracy_score(Y_test, Y_pred)
acc_train = accuracy_score(Y_train, Y_pred_train)

lloss = log_loss(Y_test, Y_proba)
lloss_train = log_loss(Y_train, Y_proba_train)

# stampo i risultati in console
print("-- Rete Neurale --")
print(f"TEST - Acc: {acc} / Loss: {lloss}")
print(f"TRAIN - Acc: {acc_train} / Loss: {lloss_train}")
print("=" * 100)

#! =======================================================================
#! RI-PARAMETRIZZARE L'ALGORITMO FINO AD UN RISULTATO OTTIMALE
#! =======================================================================


# TODO =====================================================================================================================================
# TODO L'ALGORITMO è ADDETSRATO E PRONTO PER ESSERE SALVATO E USATO PER PREDIRRE DATI NUOVI
# TODO =====================================================================================================================================
