import csv
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
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


train_file_path = '../data/data_train.csv'
validation_file_path = '../data/data_validation.csv'

x_train, y_train, _ = load_data(train_file_path)
x_validation, y_validation, _ = load_data(validation_file_path)

# Initializing the model:
knn21 = KNeighborsClassifier(n_neighbors=21)

# Train the model
knnmodel = knn21.fit(x_train, y_train)
with open('../models/knn_sklearn_model.pkl', 'wb') as model_file:
    pickle.dump((knnmodel), model_file)

# Making predictions by using pred() function
with open('../models/knn_sklearn_model.pkl', 'rb') as model_file:
    loaded_model = pickle.load(model_file)

y_pred_21 = loaded_model.predict(x_validation)
print("Predictions:")
print(y_pred_21)
print("Accuracy of KNN with k=21 is",
      accuracy_score(y_validation, y_pred_21)*100)
