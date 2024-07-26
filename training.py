import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

PIXELS = 100

def eigenvectors(T):
    # If u is an eigenvector of T'T, then v = Tu is an eigenvector of the covariance matrix S of T (as we wanted to compute)
    eigenvalues, eigenvectors = np.linalg.eig(T.T @ T)
    numPictures = T.shape[1]
    # Array to store the eigenvectors of S, each column will be an eigenvector/eigenface
    S_eigenvectors = np.zeros((PIXELS*PIXELS,numPictures))
    for i in range(numPictures):
        S_eigenvectors[:,i] = T @ eigenvectors[:,i]
        S_eigenvectors[:,i] = S_eigenvectors[:,i] / np.linalg.norm(S_eigenvectors[:,i])

    # Calculate the k-eigenvectors needed



    # Save data
    for i in range(numPictures):
        file_path = os.path.join("Eigenfaces","Eigenface" + str(i+1) + ".png")
        plt.imsave(file_path, (T @ eigenvectors[:][i]).reshape((PIXELS,PIXELS)).T, cmap='gray')
    np.savetxt('SavedData/eigenvectors.txt', S_eigenvectors)

    return S_eigenvectors
       
def TrainingSetWeights(T, S_eigenvectors):
    # Calculate the weights of the TrainingSet in the eigenface space
    numPictures = T.shape[1]
    weights_vectors = np.zeros((numPictures,numPictures))
    for i in range(numPictures):
        for k in range(numPictures):
            weights_vectors[k,i] = S_eigenvectors[:,k].T @ T[:,i]
    # Save data
    np.savetxt('SavedData/TrainingSetWeights.txt', weights_vectors)

def loadImages():
    files = os.listdir("TrainingSet")
    numPictures = len(files)
    # Array to store the images, each image is a column in T
    T = np.zeros((PIXELS*PIXELS, numPictures))
    # Load the photos
    for k in range(numPictures):
        img = Image.open("TrainingSet/" + files[k])
        img = img.resize((PIXELS,PIXELS))
        img = img.convert('L')
        I = img.load()
        l = 0
        for i in range(PIXELS):
            for j in range(PIXELS):
                T[l,k] = I[i,j]
                l += 1
    return T

def meanCenteringMatrix(T):
    m = np.mean(T,axis=1) 
    np.savetxt('SavedData/mean.txt', m)
    numPictures = T.shape[1]
    # Substract the mean to each column
    for j in range(numPictures):
        for i in range(PIXELS*PIXELS):
            T[i,j] = T[i,j] - m[i]
    return T


def main():
    T = loadImages()
    T = meanCenteringMatrix(T)
    S_eigenvectors = eigenvectors(T)
    TrainingSetWeights(T, S_eigenvectors)
    

 
if __name__ == "__main__":
    main()