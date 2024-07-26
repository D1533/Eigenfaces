import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

PIXELS = 100
DISTANCE_THRESHOLD = 3000

def readData(): 
    W = np.loadtxt('SavedData/TrainingSetWeights.txt')
    mean = np.loadtxt('SavedData/mean.txt')
    eigenvectors = np.loadtxt('SavedData/eigenvectors.txt')
    names = os.listdir("TrainingSet")
    
    return [W, mean, eigenvectors, names]

def loadTestImage(image_path):
    # Reads and processes Image to classify
    test_img = Image.open(image_path)
    test_img = test_img.resize((PIXELS,PIXELS))
    test_img = test_img.convert('L')
    I = test_img.load()

    test_image = np.zeros((PIXELS*PIXELS,1))
    l = 0
    for i in range(PIXELS):
        for j in range(PIXELS):
            test_image[l,0] = I[i,j]
            l += 1
    return test_image

def classifyImage(W, mean, eigenvectors, test_imgP, names):
    numEigenvectors = eigenvectors.shape[1]
    weights_vectorsimage = np.zeros((numEigenvectors,1))
    # Calculate weights of the image to classify
    for k in range(numEigenvectors):
        weights_vectorsimage[k,0] = eigenvectors[:,k].T @ (test_imgP[:,0] - mean)

    distances = []
    # Calculate the distances with the TrainingSet images
    for i in range(numEigenvectors):
        distances.append(np.linalg.norm(weights_vectorsimage[:,0] - W[:,i]) )

    # Classify the image 
    print(distances)
    if distances[np.argmin(distances)] <= DISTANCE_THRESHOLD:
        print(names[np.argmin(distances)].split(".")[0])
    else:
        print("Unkown person")

def main():
    [W, mean, eigenvectors, names] = readData()
    test_image = loadTestImage("Tests/test04.png")
    classifyImage(W, mean, eigenvectors, test_image, names)

 
if __name__ == "__main__":
    main()
