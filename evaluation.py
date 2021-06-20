# encoding: utf-8
import numpy as np
import pickle
from classifier import NearestNeighborClassifier

# Class label for unknown subjects in test and training data.
UNKNOWN_LABEL = -1


# Evaluation of open-set face identification.
class OpenSetEvaluation:

    def __init__(self,
                 classifier=NearestNeighborClassifier(),
                 false_alarm_rate_range=np.logspace(-3, 0, 1000, endpoint=True)):

        # The false alarm rates.
        self.false_alarm_rate_range = false_alarm_rate_range

        # Datasets (embeddings + labels) used for training and testing.
        self.train_embeddings = []
        self.train_labels = []
        self.test_embeddings = []
        self.test_labels = []

        # The evaluated classifier (see classifier.py)
        self.classifier = classifier

    # Prepare the evaluation by reading training and test data from file.
    def prepare_input_data(self, train_data_file, test_data_file):

        with open(train_data_file, 'rb') as f:
            (self.train_embeddings, self.train_labels) = pickle.load(f, encoding='bytes')
        with open(test_data_file, 'rb') as f:
            (self.test_embeddings, self.test_labels) = pickle.load(f, encoding='bytes')

    # Run the evaluation and find performance measure (identification rates) at different similarity thresholds.
    def run(self):
        #Fit the classifier on the training data.
        self.classifier.fit(self.train_embeddings,self.train_labels)
        #• Predict similarities-distance on the test data.
        print(self.test_embeddings.shape)
        #sim和label与embedding等长，一个对应一个
        prediction_label,self.similarity = self.classifier.predict_labels_and_similarities(self.test_embeddings)
        print(prediction_label.shape)
        print(self.similarity.shape)
        #能确定的label，也就是test里面非unknown的
        self.similarity_known = self.similarity[self.test_labels != UNKNOWN_LABEL]
        self.prediction_label_known = prediction_label[self.test_labels != UNKNOWN_LABEL]

        #For each false alarm rate, find a similarity threshold that yields this false alarm rate on the
        #test data and compute the corresponding identification rate.

        self.similarity_thresholds = self.select_similarity_threshold(self.similarity, self.false_alarm_rate_range)
        identification_rates = self.calc_identification_rate(self.prediction_label_known)


        # Report all performance measures.
        # Return all false alarm rates, identification rates, and similarity thresholds.
        evaluation_results = {'similarity_thresholds': self.similarity_thresholds,
                              'identification_rates': identification_rates}

        return evaluation_results

    def select_similarity_threshold(self, similarity, false_alarm_rate):

        return np.percentile(similarity[self.test_labels==UNKNOWN_LABEL],(1-false_alarm_rate)*100)

    def calc_identification_rate(self, prediction_labels):

        res = []
        test_labels_known = self.test_labels[self.test_labels==UNKNOWN_LABEL]
        #每个th都对应一个acc，th其实就是rate
        for similarity_known_threshold in self.similarity_thresholds:
            n_true = 0
            for prediction_label, test_label, similarity_known in zip(prediction_labels, test_labels_known, self.similarity_known):
                if (test_label == UNKNOWN_LABEL and similarity_known > similarity_known_threshold):
                    n_true += 1

                elif prediction_label == test_label:
                    n_true += 1
            acc = n_true / len(prediction_labels)
            res.append(acc)
        return res