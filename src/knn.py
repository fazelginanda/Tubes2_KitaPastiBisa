import csv
import math
import numpy as np
import pickle


def load_data(file_path):
    features = []
    labels = []
    with open(file_path, 'r') as csv_file:
        csv_reader = csv.reader(csv_file)
        header = next(csv_reader)
        for row in csv_reader:
            features.append([float(val) if '.' in val else int(val)
                            for val in row[:-1]])
            labels.append(row[-1])
    return features, labels, header[:-1]


def load_data_model(file_path):
    data = []
    with open(file_path, 'r') as csv_file:
        csv_reader = csv.reader(csv_file)
        for row in csv_reader:
            data.append([float(val) if '.' in val else int(val)
                         for val in row])
    return data


def get_min_of_column(data, index):
    column_values = []
    for row in data:
        column_values.append(row[index])
    return min(column_values)


def get_max_of_column(data, index):
    column_values = []
    for row in data:
        column_values.append(row[index])
    return max(column_values)


def normalize_data(data):
    for row in data:
        for i in range(len(row)):
            row[i] = (row[i] - get_min_of_column(data, i)) / \
                (get_max_of_column(data, i) - get_min_of_column(data, i))
    return data


def euclidean_distance(row1, row2):
    sum_of_squared_diff = 0
    for i in range(len(row1)):
        sum_of_squared_diff += (row2[i]-row1[i]) ** 2
    return math.sqrt(sum_of_squared_diff)


def get_neighbors(train_data, test_feature_row, num_of_neighbors):
    distances = []
    for train_row in train_data:
        train_feature_row = train_row[:-1]
        distance = euclidean_distance(train_feature_row, test_feature_row)
        distances.append((train_feature_row, distance))
        distances.sort(key=lambda x: x[1])
    neighbors = []
    for i in range(num_of_neighbors):
        neighbors.append(distances[i][0])
    return neighbors


def predict(train_data, test_feature_row, num_of_neighbors):
    neighbors = get_neighbors(train_data, test_feature_row, num_of_neighbors)
    list_of_predicted_labels = []
    for neighbor in neighbors:
        list_of_predicted_labels.append(neighbor[-1])
    prediction = max(set(list_of_predicted_labels),
                     key=list_of_predicted_labels.count)
    return prediction


def evaluate_model(train_data, test_feature_data, actual_labels):
    predictions = []
    for instance in test_feature_data:
        predictions.append(predict(train_data, instance, 9))

    accuracy = sum(pred == actual for pred, actual in zip(
        predictions, actual_labels)) / len(predictions)
    print("Accuracy:", accuracy)


train_file_path = './data/data_train.csv'
test_file_path = './data/data_validation.csv'

# train_data = load_data_model(train_file_path)
train_features, train_labels, _ = load_data(test_file_path)
test_features, actual_labels, _ = load_data(test_file_path)

train_data = []
for i in range(len(train_features)):
    train_data.append(train_features[i])
    train_data[i].append(train_labels[i])

evaluate_model(train_data, test_features, actual_labels)


##### TESTING #####
# for row in train_data:
#     print(row)

# def print_matrix(matrix):
#     for row in matrix:
#         print(row)

# def get_num_of_instance(data):
#     return len(data)
