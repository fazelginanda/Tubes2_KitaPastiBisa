import csv
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier


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


train_file_path = './data/data_train.csv'
validation_file_path = './data/data_validation.csv'

x_train, y_train, _ = load_data(train_file_path)
x_validation, y_validation, _ = load_data(validation_file_path)

knn1 = KNeighborsClassifier(n_neighbors=1)
knn3 = KNeighborsClassifier(n_neighbors=3)
knn5 = KNeighborsClassifier(n_neighbors=5)
knn7 = KNeighborsClassifier(n_neighbors=7)
knn9 = KNeighborsClassifier(n_neighbors=9)

knn1.fit(x_train, y_train)
knn3.fit(x_train, y_train)
knn5.fit(x_train, y_train)
knn7.fit(x_train, y_train)
knn9.fit(x_train, y_train)

y_pred_1 = knn1.predict(x_validation)
y_pred_3 = knn3.predict(x_validation)
y_pred_5 = knn5.predict(x_validation)
y_pred_7 = knn7.predict(x_validation)
y_pred_9 = knn9.predict(x_validation)

print("Accuracy with k=1", accuracy_score(y_validation, y_pred_1)*100)
print("Accuracy with k=3", accuracy_score(y_validation, y_pred_3)*100)
print("Accuracy with k=5", accuracy_score(y_validation, y_pred_5)*100)
print("Accuracy with k=7", accuracy_score(y_validation, y_pred_7)*100)
print("Accuracy with k=9", accuracy_score(y_validation, y_pred_9)*100)
