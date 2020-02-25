"""
Perceptron test code
"""
import re
import os
import json
import sys

model_path = sys.argv[1]
test_input_path = sys.argv[2]


def PerceptronTest():
    test_data = Documents()
    test_data.read_json()
    test_data.parse_test_input()
    test_data.classify()


class Documents:

    def __init__(self):
        self.test_data = {}
        self.idf = {}
        self.weights_truth = {}
        self.weights_pos = {}
        self.bias_truth = 0
        self.bias_pos = 0
        self.class_names = ['positive_truthful', 'negative_truthful', 'positive_deceptive', 'negative_deceptive']

    def parse_test_input(self):
        fp_out = open('percepoutput.txt', 'w')
        fp_out.close()
        file_count = 0
        for (root, dirs, files) in os.walk(test_input_path, topdown=True):
            for file in files:
                if file.startswith('.') or 'README' in file or (not file.endswith('.txt')):
                    continue
                file_count += 1
                file_path = root + '/' + file
                fp = open(file_path, 'r')
                temp_line = re.sub(r"[^a-zA-Z?!']+", ' ', ''.join(fp.readlines()).strip())
                self.test_data[file_path] = {}

                for word in temp_line.split():
                    word = word.lower()

                    if word in self.test_data[file_path]:
                        self.test_data[file_path][word] += 1
                    else:
                        self.test_data[file_path][word] = 1

    def read_json(self):
        with open(model_path, 'r') as json_file:
            temp_list = json.load(json_file)
            self.weights_truth = temp_list[0]
            self.weights_pos = temp_list[1]
            self.bias_truth = temp_list[2]['Bias Truth']
            self.bias_pos = temp_list[2]['Bias Positive']

    def classify(self):

        fp_out = open('percepoutput.txt', 'a')
        for file in self.test_data:
            value_truth = 0
            value_positive = 0
            for word in self.test_data[file]:
                if word in self.weights_truth:
                    value_truth += self.weights_truth[word] * self.test_data[file][word]

                if word in self.weights_pos:
                    value_positive += self.weights_pos[word] * self.test_data[file][word]

            value_positive += self.bias_pos
            value_truth += self.bias_truth

            if value_truth > 0:
                output_string = 'truthful '
            else:
                output_string = 'deceptive '
            if value_positive > 0:
                output_string += 'positive '
            else:
                output_string += 'negative '

            # Write result to output file
            fp_out.write(output_string + file + '\n')
        fp_out.close()


if __name__ == '__main__':
    PerceptronTest()
