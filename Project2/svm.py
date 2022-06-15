import argparse
import os
import tqdm

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import cv2

import sklearn as sk
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from mpl_toolkits.mplot3d import Axes3D

# Helper function for plotting the fit of your SVM.
def plot_fit(X, y, clf):
    """
    X = samples
    y = Ground truth
    clf = trained model
    """
    h = .02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    fig = plt.figure(1, figsize=(8, 6))
    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)

    # Plot also the training points
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, edgecolors= "black")
    plt.xlabel('Sepal length')
    plt.ylabel('Sepal width')
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xticks(())
    plt.yticks(())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--c", type=float, default=0.01)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--batchsize", type=int, default=1)
    parser.add_argument("--epoch", type=int, default=10)
    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    parser.add_argument("--input", type=str, default="./input")
    parser.add_argument("--output", type=str, default="./output")
    parser.add_argument("--factor", type=float, default=2)
    args = parser.parse_args()
    args.device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    #Load data: 1=blonde, 0=brunette
    X = []
    Y = []
    names = []
    for folder in os.listdir(args.input):
        for filename in os.listdir(os.path.join(args.input, folder)):
            names.append(filename)
            print(os.path.join(args.input, folder, filename))
            latent = np.load(os.path.join(args.input, folder, filename))['w']
            X.append(latent.flatten())
            if folder == 'blondes':
                Y.append(1)
            else:
                Y.append(0)
    X = np.array(X)
    Y = np.array(Y)

    print(type(X), X.shape)
    print(type(Y), Y.shape)

    # Initialize classifier 
    svc = sk.svm.SVC(kernel="linear")

    # Fit model and print accuracy score
    svc.fit(X, Y)
    score_lsvc = svc.score(X, Y)
    print(score_lsvc)
    print(svc.support_vectors_.shape)

    for space in range(27):
        # print('b = ',svc.intercept_)
        # print('Indices of support vectors = ', svc.support_)
        # print('Support vectors = ', svc.support_vectors_)
        # print('Number of support vectors for each class = ', svc.n_support_)
        # print('Coefficients of the support vector in the decision function = ', np.abs(svc.dual_coef_))
        # break
        print(svc.support_vectors_[space])
        PATATA = np.reshape(svc.support_vectors_[space], (1,18,512))
        name = names[space].replace('.npz', '')
        print(f'{args.output}/{name}.npy')
        np.save(f'{args.output}/{name}.npy', PATATA)