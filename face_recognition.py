import cv2
import numpy as np
import pickle
import os
import random
from scipy import spatial
from collections import Counter
from sklearn.preprocessing import normalize

from sklearn.cluster import MiniBatchKMeans


# FaceNet to extract face embeddings.
class FaceNet:

    def __init__(self):
        self.dim_embeddings = 128
        self.facenet = cv2.dnn.readNetFromONNX("resnet50_128.onnx")

    # Predict embedding from a given face image.
    def predict(self, face):
        # Normalize face image using mean subtraction.
        face = face - (131.0912, 103.8827, 91.4953)
        # reshaped = np.moveaxis(face, 2, 0)
        #
        face=np.transpose(face,(2,0,1))
        reshaped=np.expand_dims(face,axis=0)
        print(reshaped.shape)
        # Forward pass through deep neural network. The input size should be 224 x 224.
        # reshaped = np.reshape(reshaped, (1, 3, 224, 224))
        self.facenet.setInput(reshaped)
        embedding = np.squeeze(self.facenet.forward())
        return embedding / np.linalg.norm(embedding)

    # Get dimensionality of the extracted embeddings.
    def get_embedding_dimensionality(self):
        return self.dim_embeddings


# The FaceRecognizer model enables supervised face identification.
class FaceRecognizer:

    # Prepare FaceRecognizer; specify all parameters for face identification.
    def __init__(self, num_neighbours=21, max_distance=1.01, min_prob=0.8):
        # ToDo: Prepare FaceNet and set all parameters for kNN.
        self.k = num_neighbours
        self.max_distance = max_distance
        self.min_prob = min_prob
        self.facenet = FaceNet()
        self.loaded=False
        # The underlying gallery: class labels and embeddings.
        self.labels = []
        self.embeddings = np.empty((0, self.facenet.get_embedding_dimensionality()))
        self.tmp = np.empty((0, self.facenet.get_embedding_dimensionality()))

        # Load face recognizer from pickle file if available.
        if os.path.exists("recognition_gallery.pkl"):
            self.load()

    # Save the trained model as a pickle file.
    def save(self):
        with open("recognition_gallery.pkl", 'wb') as f:
            pickle.dump((self.labels, self.embeddings), f)

    # Load trained model from a pickle file.
    def load(self):
        self.loaded=True
        print("loaded")
        with open("recognition_gallery.pkl", 'rb') as f:
            (self.labels, self.embeddings) = pickle.load(f)

    # ToDo
    # def update(self, face, label):
    #     if label in   self.labels:
    #         pass
    #     else:
    #         self.labels.append(label)
    #     self.tmp=np.concatenate((self.tmp, self.facenet.predict(face).reshape((1,-1))),axis=0)
    #     self.tmp = np.mean(self.tmp,axis=0)
    #     self.tmp = np.expand_dims(self.tmp,axis=0)

    def update(self, face, label):
        self.labels.append(label)
        self.embeddings = np.append(self.embeddings, [self.facenet.predict(face)], axis=0)

        # norm=self.facenet.predict(face).reshape((1, -1))
        # self.embeddings = np.concatenate((self.embeddings,norm ), axis=0)

    def opt(self):
        if self.loaded:
            self.embeddings = np.concatenate((self.embeddings,self.tmp),axis=0)
        else:
            self.embeddings=self.tmp


    # ToDo return 3 value: label ,prob ,distance

    def predict(self, face):
        bf = cv2.BFMatcher()
        # create hard assignment
        # temp = np.zeros((1, self.facenet.get_embedding_dimensionality() ))
        # temp[0] = self.facenet.predict(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))

        input=self.facenet.predict(face).reshape((1,-1)).astype(np.float32)
        input=np.repeat(input,len(self.embeddings),axis=0)
        # distance=np.linalg.norm((self.embeddings-input),axis=1)
        # idx=np.argsort(distance)[:21]
        # label=[]
        # distancedict={}
        #
        # for i in idx:
        #     if self.labels[i] in label:
        #         if distancedict[self.labels[i]] > distance[i]:
        #             distancedict[self.labels[i]]=distance[i]
        #     else:
        #         distancedict.setdefault(self.labels[i],distance[i])
        #
        #     label.append(self.labels[i])
        #
        # predicted_label=Counter(label).most_common(1)[0][0]
        # dist_to_prediction=distancedict[predicted_label]
        # prob=Counter(label).most_common(1)[0][1]/self.k
        #
        # if self.max_distance <= dist_to_prediction or self.min_prob > prob:
        #     return "unknown", prob, dist_to_prediction
        # return predicted_label, prob, dist_to_prediction
        matches = bf.knnMatch(input.astype(np.float32),self.embeddings.astype(np.float32),k=self.k)

        label=[]
        labeldict={}
        distancedict={}
        for idx, m in enumerate(matches[0]):
            if self.labels[m.trainIdx] in label:
                labeldict[self.labels[m.trainIdx]]=labeldict[self.labels[m.trainIdx]]+1
                if m.distance < distancedict[self.labels[m.trainIdx]+'dis']:
                    distancedict[self.labels[m.trainIdx]+'dis']=m.distance
            else:
                labeldict.setdefault(self.labels[m.trainIdx],1)
                distancedict.setdefault(self.labels[m.trainIdx]+'dis',m.distance)
            label.append(self.labels[m.trainIdx])
            print(m.trainIdx)
            print(m.distance)
            print(self.labels[m.trainIdx])
            print('-' * 20)
        print('*'*10)
        labeldict=sorted(labeldict.items(), key=lambda kv: (kv[1], kv[0]),reverse=True)
        prob = labeldict[0][1] / self.k
        tmp = labeldict[0][0] + 'dis'

        dist_to_prediction = distancedict[tmp]

        predicted_label = Counter(label).most_common(1)[0][0]

        if self.max_distance <= dist_to_prediction or self.min_prob > prob:
            return "unknown", prob, dist_to_prediction
        return predicted_label, prob, dist_to_prediction



# The FaceClustering class enables unsupervised clustering of face images according to their identity and
# re-identification.
class FaceClustering:

    # Prepare FaceClustering; specify all parameters of clustering algorithm.
    def __init__(self,num_clusters=2, max_iter=25):
        # ToDo: Prepare FaceNet.
        self.facenet = FaceNet()
        # The underlying gallery: embeddings without class labels.
        self.embeddings = np.empty((0, self.facenet.get_embedding_dimensionality()))

        # Number of cluster centers for k-means clustering.
        self.num_clusters = num_clusters
        # Cluster centers.
        self.cluster_center = np.empty((num_clusters, self.facenet.get_embedding_dimensionality()))
        # Cluster index associated with the different samples.
        self.cluster_membership = []

        # Maximum number of iterations for k-means clustering.
        self.max_iter = max_iter

        # Load face clustering from pickle file if available.
        if os.path.exists("clustering_gallery.pkl"):
            self.load()

    # Save the trained model as a pickle file.
    def save(self):
        with open("clustering_gallery.pkl", 'wb') as f:
            pickle.dump((self.embeddings, self.num_clusters, self.cluster_center, self.cluster_membership), f)

    # Load trained model from a pickle file.
    def load(self):
        with open("clustering_gallery.pkl", 'rb') as f:
            (self.embeddings, self.num_clusters, self.cluster_center, self.cluster_membership) = pickle.load(f)

    # ToDo
    def update(self, face):
        self.embeddings = np.concatenate((self.embeddings, self.facenet.predict(face).reshape((1, -1))), axis=0)

    # ToDo
    def fit(self):

        dummy = MiniBatchKMeans(n_clusters=self.num_clusters, batch_size=3000, random_state=9).fit(self.embeddings)
        self.cluster_center = dummy.cluster_centers_
        self.cluster_membership = dummy.labels_

    # ToDo
    def predict(self, face):
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(self.facenet.predict(face).reshape((1, -1)).astype(np.float32),
                              self.embeddings.astype(np.float32), k=self.num_clusters)
        distance = []
        label = []

        for idx, m in enumerate(matches[0]):
            distance.append(m.distance)
            print(m.trainIdx)
            label.append(self.cluster_membership[m.trainIdx])
        idx = np.where(distance == np.min(distance))[0]

        return self.cluster_center[idx], np.min(distance)
