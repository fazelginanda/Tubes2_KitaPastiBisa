# Importing the necessary packages
import csv
import pickle
import sklearn
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

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

# Let's initializing the model:
NBclassifier = GaussianNB()

# Train the model:
NBmodel = NBclassifier.fit(x_train, y_train)
with open('./models/naive_bayes_sklearn_model.pkl', 'wb') as model_file:
        pickle.dump((NBmodel), model_file)

# Making predictions by using pred() function:
with open('./models/naive_bayes_sklearn_model.pkl', 'rb') as model_file:
        loaded_model = pickle.load(model_file)

NBpreds = loaded_model.predict(x_validation)
# print("The predictions are:\n", NBpreds)

# Finding accuracy of our Naive Bayes classifier:
print("Accuracy of our classifier is:", accuracy_score(y_validation, NBpreds) *100)
