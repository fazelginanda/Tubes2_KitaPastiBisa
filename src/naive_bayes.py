import csv
import math
import numpy as np
from collections import defaultdict
import pickle

def load_data(file_path):
    features = []
    labels = []
    with open(file_path, 'r') as csv_file:
        csv_reader = csv.reader(csv_file)
        header = next(csv_reader)
        for row in csv_reader:
            features.append([float(val) if '.' in val else int(val) for val in row[:-1]])
            labels.append(row[-1])
    return features, labels, header[:-1]

def separate_by_class(features, labels):
    separated = defaultdict(list)
    for feat, label in zip(features, labels):
        separated[label].append(feat)
    return separated

def summarize_data(data):
    summaries = {}
    for class_label, instances in data.items():
        means = [sum(x) / len(x) for x in zip(*instances)]
        if all(isinstance(val, (int, float)) for val in instances[0]):
            is_categorical = [len(set(instances[i])) <= 2 for i in range(len(instances))]
            if is_categorical:
                probabilities = [sum(x) / len(x) for x in zip(*instances)]
                summaries[class_label] = {'prob': probabilities, 'is_categorical': True}
            else:
                std_devs = [math.sqrt(sum((x - mean)**2 for x in instances[i]) / (len(instances[i])-1)) for i, mean in enumerate(means)]
                summaries[class_label] = {'mean': means, 'std_dev': std_devs, 'is_categorical': False}
    return summaries

# gaussian probability for numerical features
def calculate_numerical_probability(x, mean, stdev):
    exponent = math.exp(-(math.pow(x - mean, 2) / (2 * math.pow(stdev, 2))))
    return (1 / (math.sqrt(2 * math.pi) * stdev)) * exponent

# bernoulli for categorical features
def calculate_categorical_probability(x, prob):
    prob = np.clip(prob, 1e-15, 1 - 1e-15)
    return np.exp(x * np.log(prob) + (1 - x) * np.log(1 - prob))

def calculate_class_probabilities(summaries, input_vector):
    probabilities = {}
    for class_label, class_summaries in summaries.items():
        probabilities[class_label] = 1
        for i in range(len(input_vector)):
            if class_summaries['is_categorical']:
                prob = class_summaries['prob'][i]
                x = input_vector[i]
                probabilities[class_label] *= calculate_categorical_probability(x, prob)
            else:
                mean = class_summaries['mean'][i]
                stdev = class_summaries['std_dev'][i]
                x = input_vector[i]
                probabilities[class_label] *= calculate_numerical_probability(x, mean, stdev)
    return probabilities

# predict class for new instance
def predict(summaries, input_vector):
    probabilities = calculate_class_probabilities(summaries, input_vector)
    best_label, best_prob = None, -1
    for class_label, probability in probabilities.items():
        if best_label is None or probability > best_prob:
            best_prob = probability
            best_label = class_label
    return best_label

def train_naive_bayes(train_file_path):
    # lad training data
    features, labels, header = load_data(train_file_path)

    # separate data by class
    separated_data = separate_by_class(features, labels)

    # summarize data by class
    summaries = summarize_data(separated_data)

    # save the model
    with open('./models/naive_bayes_model.pkl', 'wb') as model_file:
        pickle.dump((summaries, header), model_file)

def load_naive_bayes_model(model_file_path):
    # load the model
    with open(model_file_path, 'rb') as model_file:
        summaries, header = pickle.load(model_file)
    return summaries, header

def evaluate_model(model, validation_data, actual_labels):
    predictions = [predict(model, instance) for instance in validation_data]

    # accuracy
    accuracy = sum(pred == actual for pred, actual in zip(predictions, actual_labels)) / len(predictions)
    print("Accuracy:", accuracy)

# load path
train_file_path = './data/data_train.csv'
validation_file_path = './data/data_validation.csv'

# load validation data
validation_features, actual_labels, _ = load_data(validation_file_path)

# train
train_naive_bayes(train_file_path)

# load model
loaded_model, loaded_header = load_naive_bayes_model('./models/naive_bayes_model.pkl')

# evaluate model
evaluate_model(loaded_model, validation_features, actual_labels)
