"""
Perceptron training code
"""
import re
import os
import math
import json
import sys
import statistics
import numpy as np

train_dir = sys.argv[1]

SD_FACTOR = 2
MAX_ITERATIONS = 30


def PerceptronTrain():
    train_data = Documents()
    train_data.parse_input()
    train_data.calculate_tfidf()
    train_data.learn_model_weights()


class Documents:

    def __init__(self):
        self.vocabulary = set()
        self.positive_truthful = {}
        self.negative_truthful = {}
        self.positive_deceptive = {}
        self.negative_deceptive = {}
        self.idf = {}
        self.weights = {}
        self.tfidf_list = []
        self.remove_words = set()
        self.class_truth = None
        self.class_positive = None
        self.weights_truth = None
        self.weights_pos = None
        self.file_count = 0
        self.bias_truth = 0
        self.bias_pos = 0
        self.word_count_vector = None
        self.word_count = 0
        self.tfidf = {'positive_truthful': {}, 'negative_truthful': {}, 'positive_deceptive': {},
                      'negative_deceptive': {}}
        self.class_frequency = [self.positive_truthful, self.negative_truthful, self.positive_deceptive,
                                self.negative_deceptive]
        self.class_names = ['positive_truthful', 'negative_truthful', 'positive_deceptive', 'negative_deceptive']

    def parse_input(self):
        index = -1
        for (root, dirs, files) in os.walk(train_dir, topdown=True):

            if re.search(r".*positive.*truthful.*", root):
                index = 0
            elif re.search(r".*negative.*truthful.*", root):
                index = 1
            elif re.search(r".*positive.*deceptive.*", root):
                index = 2
            elif re.search(r".*negative.*deceptive.*", root):
                index = 3

            if index == -1:
                continue

            for file in files:
                if file.startswith('.') or 'README' in file:
                    continue
                self.file_count += 1
                file_path = root + '/' + file
                fp = open(file_path, 'r')
                temp_line = re.sub(r"[^a-zA-Z']+", ' ', ''.join(fp.readlines()).strip())
                self.class_frequency[index][file_path] = {}

                for word in temp_line.split():
                    word = word.lower()
                    self.word_count += 1
                    if word in self.class_frequency[index][file_path]:
                        self.class_frequency[index][root + '/' + file][word] += 1
                    else:
                        self.class_frequency[index][file_path][word] = 1
                        if word in self.idf:
                            self.idf[word] += 1
                        else:
                            self.idf[word] = 1
                    self.vocabulary.add(word)

        # Calculate idf for each word
        for word, val in self.idf.items():
            self.idf[word] = math.log10(self.word_count) - math.log10(val + 1)

    def shuffle_vectors(self):
        index_shuffle = np.random.permutation(self.class_positive.shape[0])
        self.class_positive = self.class_positive[index_shuffle]
        self.class_truth = self.class_truth[index_shuffle]
        self.word_count_vector = self.word_count_vector[index_shuffle]

    def learn_model_weights(self):
        self.weights_truth = np.zeros((len(self.vocabulary),))
        self.weights_pos = np.zeros((len(self.vocabulary),))
        self.vectorize_documents()

        # Averaged model
        cached_wtruth = np.zeros((len(self.vocabulary),))
        cached_wpos = np.zeros((len(self.vocabulary),))
        cached_beta_truth = 0
        cached_beta_pos = 0
        count = 1

        for _ in range(MAX_ITERATIONS):
            self.shuffle_vectors()
            index = 0
            for row in self.word_count_vector:
                activation_truth = np.dot(row, self.weights_truth)
                activation_pos = np.dot(row, self.weights_pos)

                activation_truth = np.add(activation_truth, self.bias_truth)
                activation_pos = np.add(activation_pos, self.bias_pos)

                if (self.class_truth[index] * activation_truth) <= 0:
                    self.weights_truth = np.add(self.weights_truth, np.dot(row, self.class_truth[index]))
                    self.bias_truth = np.add(self.bias_truth, self.class_truth[index])

                    # Averaged model
                    cached_wtruth = np.add(cached_wtruth, np.dot(count, np.dot(row, self.class_truth[index])))
                    cached_beta_truth = np.add(cached_beta_truth, count * self.class_truth[index])

                if (self.class_positive[index] * activation_pos) <= 0:
                    self.weights_pos = np.add(self.weights_pos, np.dot(row, self.class_positive[index]))
                    self.bias_pos = np.add(self.bias_pos, self.class_positive[index])

                    # Averaged model
                    cached_wpos = np.add(cached_wpos, np.dot(count, np.dot(row, self.class_positive[index])))
                    cached_beta_pos = np.add(cached_beta_pos, count * self.class_positive[index])

                index += 1
                count += 1

        # Write model parameters for vanilla model
        self.write_model('vanillamodel.txt')

        # Calculate for averaged model
        self.weights_truth = np.subtract(self.weights_truth, np.divide(cached_wtruth, count))
        self.weights_pos = np.subtract(self.weights_pos, np.divide(cached_wpos, count))

        self.bias_pos -= cached_beta_pos / count
        self.bias_truth -= cached_beta_truth / count

        #Write model parameters for averaged model
        self.write_model('averagedmodel.txt')

    def vectorize_documents(self):
        temp_vector = []
        cls_truth = []
        cls_positive = []
        temp_count_vector = []
        for index, class_freq in enumerate(self.class_frequency):
            for file in class_freq:
                cls_truth.append(1 if 'truth' in file else -1)
                cls_positive.append(1 if 'positive' in file else -1)
                for word in self.vocabulary:
                    if word in self.class_frequency[index][file]:
                        temp_vector.append(self.tfidf[self.class_names[index]][file][word])
                        temp_count_vector.append(self.class_frequency[index][file][word])
                    else:
                        temp_vector.append(0)
                        temp_count_vector.append(0)

        self.word_count_vector = np.array(temp_count_vector).reshape((self.file_count, len(self.vocabulary)))
        self.class_truth = np.array(cls_truth)
        self.class_positive = np.array(cls_positive)

    def calculate_tfidf(self):
        for index, class_freq in enumerate(self.class_frequency):
            for file in class_freq:
                tf = {}
                self.tfidf[self.class_names[index]][file] = {}
                total = sum(class_freq[file].values())
                for word in class_freq[file]:
                    tf[word] = math.log10(class_freq[file][word]) - math.log10(total) + 1
                    self.tfidf[self.class_names[index]][file][word] = tf[word] * self.idf[word]
                self.tfidf_list.extend(list(self.tfidf[self.class_names[index]][file].values()))

        mean = statistics.mean(self.tfidf_list)
        sd = statistics.pstdev(self.tfidf_list)

        for index, class_freq in enumerate(self.class_frequency):

            for file in class_freq:
                for word in class_freq[file]:
                    if self.tfidf[self.class_names[index]][file][word] < (mean - SD_FACTOR * sd):
                        self.remove_words.add(word)

        for index, class_freq in enumerate(self.class_frequency):
            for file in class_freq:
                for word in self.remove_words:
                    if word in self.class_frequency[index][file]:
                        self.class_frequency[index][file].pop(word)

    def write_model(self, model):

        fp = open(model, 'w')
        weights_t = {}
        weights_p = {}
        for index, word in enumerate(self.vocabulary):
            weights_t[word] = self.weights_truth[index]
            weights_p[word] = self.weights_pos[index]
        bias = dict()
        bias['Bias Truth'] = int(self.bias_truth)
        bias['Bias Positive'] = int(self.bias_pos)
        temp_list = [weights_t, weights_p, bias]
        json.dump(temp_list, fp)
        fp.close()


if __name__ == '__main__':
    PerceptronTrain()
