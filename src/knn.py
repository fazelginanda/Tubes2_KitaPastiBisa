import numpy as np
import pickle

class KNNClassifier:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        predictions = [self._predict(x) for x in X_test]
        return np.array(predictions)

    def _predict(self, x):
        distances = [np.linalg.norm(x - x_train) for x_train in self.X_train]
        k_neighbors_indices = np.argsort(distances)[:self.k]
        k_neighbor_labels = [self.y_train[i] for i in k_neighbors_indices]
        most_common = np.bincount(k_neighbor_labels).argmax()
        return most_common

def load_data_from_csv(file_path):
    data = np.genfromtxt(file_path, delimiter=',', skip_header=1)
    X_train = data[:, :-1]  # kolom-kolom fitur
    y_train = data[:, -1]   # kolom target
    return X_train, y_train

def calculate_accuracy(y_true, y_pred):
    correct_predictions = np.sum(y_true == y_pred)
    total_samples = len(y_true)
    accuracy = correct_predictions / total_samples
    return accuracy

# loading data train
training_file_path = '../data/data_train.csv'
X_train, y_train = load_data_from_csv(training_file_path)

# loading data test
test_file_path = '../data/data_validation.csv'
data_test = np.genfromtxt(test_file_path, delimiter=',', skip_header=1)
X_test = data_test[:, :-1]
Y_test = data_test[:, -1]

# create and fit the k-NN classifier
knn_classifier = KNNClassifier(k=21)
knn_classifier.fit(X_train, y_train)
with open('../models/knn_model.pkl', 'wb') as model_file:
        pickle.dump((knn_classifier), model_file)

# making predictions by using pred() function:
with open('../models/knn_model.pkl', 'rb') as model_file:
        loaded_model = pickle.load(model_file)

# make predictions
predictions = loaded_model.predict(X_test)
accuracy = calculate_accuracy(Y_test, predictions)

print("Predictions:")
print(predictions)
print(f"Accuracy: {accuracy*100}%")