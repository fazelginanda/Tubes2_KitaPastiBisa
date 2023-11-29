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
            labels.append(int(row[-1]))
    return features, labels, header

# Separate and collect rows by the price_range target categories (0, 1, 2, 3)
def separate_by_class(features, labels):
    separated = defaultdict(list)
    for feat, label in zip(features, labels):
        separated[label].append(feat)
    return separated

def summarize_data(data):
    numerical = {}
    categorical = {}
    
    for class_label, instances in data.items():
        means = []
        std_devs = []
        prob_0 = []
        prob_1 = []
        rows = len(instances)
        for j in range(20):
            if (j in (1, 3, 5, 17, 18, 19)): # Categorical data (non numeric)
                count_0 = 0
                count_1 = 0
                for i in range(rows):
                    if (instances[i][j] == 0):
                        count_0 += 1
                    else:
                        count_1 += 1
                prob_0.append((float(count_0) / rows)) # probabilities of 0 in categorical columns
                prob_1.append((float(count_1) / rows)) # probabilities of 1 in categorical columns
                    
            else: # Numerical data
                sum = 0
                sum_squared_diff = 0
                # Calculating mean for numerical data
                for i in range(rows):
                    sum += instances[i][j]
                mean = float(sum) / rows
                means.append(mean)
                
                # Calculating STD for numerical data
                for i in range(rows):
                    sum_squared_diff += (instances[i][j] - mean) ** 2
                mean_squared_diff = float(sum_squared_diff) / rows
                std_dev = math.sqrt(mean_squared_diff)
                std_devs.append(std_dev)

        numerical[class_label] = {'mean' : means, 'std_dev' : std_devs}
        categorical[class_label] = {'prob_0' : prob_0, 'prob_1' : prob_1}
    return numerical, categorical

# gaussian probability for numerical features
def calculate_numerical_probability(x, mean, stdev):
    exponent = math.exp(-(math.pow(x - mean, 2) / (2 * math.pow(stdev, 2))))
    return (exponent / (math.sqrt(2 * math.pi) * stdev))

# bernoulli for categorical features
def calculate_categorical_probability(x, prob):
    prob = np.clip(prob, 1e-15, 1 - 1e-15)
    return np.exp(x * np.log(prob) + (1 - x) * np.log(1 - prob))

def calculate_class_probabilities(summaries, input):
    probabilities = {}
    
    for class_label, class_summaries in summaries[0].items():
        probabilities[class_label] = 1
        j = 0
        for i in range(len(input)):
            if (i not in (1, 3, 5, 17, 18, 19)):
                mean = class_summaries['mean'][j]
                stdev = class_summaries['std_dev'][j]
                x = input[i]
                probabilities[class_label] *= calculate_numerical_probability(x, mean, stdev)
                j += 1
                
    for class_label, class_summaries in summaries[1].items():
        # probabilities[class_label] = 1
        j = 0
        for i in range(len(input)):
            if (i in (1, 3, 5, 17, 18, 19)):
                prob = class_summaries['prob_1'][j]
                x = input[i]
                probabilities[class_label] *= calculate_categorical_probability(x, prob)
                j += 1
    return probabilities

# predict class for new instance
def predict(summaries, input):
    probabilities = calculate_class_probabilities(summaries, input)
    best_label, best_prob = None, -1
    for class_label, probability in probabilities.items():
        if best_label is None or probability > best_prob:
            best_prob = probability
            best_label = class_label
    return best_label

def train_naive_bayes(train_file_path):
    # load training data
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

def evaluate_model(model, validation_data, target_labels):
    predictions = [predict(model, instance) for instance in validation_data]

    # accuracy
    accuracy = sum(pred == target for pred, target in zip(predictions, target_labels)) / len(predictions)
    print("Accuracy:", accuracy)

# load path
train_file_path = './data/data_train.csv'
validation_file_path = './data/data_validation.csv'

# load validation data
validation_features, target_labels, header = load_data(validation_file_path)

# train
train_naive_bayes(train_file_path)

# load model
loaded_model, loaded_header = load_naive_bayes_model('./models/naive_bayes_model.pkl')

# evaluate model
evaluate_model(loaded_model, validation_features, target_labels)
