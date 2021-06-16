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
        prediction_label,self.similarity = self.classifier.predict_labels_and_similarities(self.test_embeddings)


        #For each false alarm rate, find a similarity threshold that yields this false alarm rate on the
        #test data and compute the corresponding identification rate.
        self.similarity_thresholds = self.select_similarity_threshold(self.similarity, self.false_alarm_rate_range)
        print(len(self.similarity))
        print("hi：",self.similarity)
        identification_rates = self.calc_identification_rate(prediction_label)


        # Report all performance measures.
        # Return all false alarm rates, identification rates, and similarity thresholds.
        evaluation_results = {'similarity_thresholds': self.similarity_thresholds,
                              'identification_rates': identification_rates}

        return evaluation_results

    def select_similarity_threshold(self, similarity, false_alarm_rate):

        return np.percentile(similarity,false_alarm_rate*100)

    def calc_identification_rate(self, prediction_labels):

        res = []
        for threshold in self.similarity_thresholds:
            n_true = 0
            for prediction_label, test_label, s in zip(prediction_labels, self.test_labels, self.similarity):
                # print(s,t)
                if prediction_label == test_label and s >= threshold:
                    n_true += 1
            acc = n_true / len(prediction_labels)
            res.append(acc)
        return res