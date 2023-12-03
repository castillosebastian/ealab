from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import RidgeClassifier, SGDClassifier
from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC, NuSVC, SVC
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.metrics import classification_report

# Load Iris dataset
data = load_iris()
X, y = data.data, data.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the classifiers
classifiers = {
    "LDA": LinearDiscriminantAnalysis(),
    "QDA": QuadraticDiscriminantAnalysis(),
    "AdaBoost": AdaBoostClassifier(),
    "Bagging": BaggingClassifier(),
    "Extra Trees Ensemble": ExtraTreesClassifier(),
    "Gradient Boosting": GradientBoostingClassifier(),
    "Random Forest": RandomForestClassifier(),
    "Ridge": RidgeClassifier(),
    "SGD": SGDClassifier(),
    "BNB": BernoulliNB(),
    "GNB": GaussianNB(),
    "KNN": KNeighborsClassifier(),
    "MLP": MLPClassifier(),
    "LSVC": LinearSVC(),
    "NuSVC": NuSVC(),
    "SVC": SVC(),
    "DTC": DecisionTreeClassifier(),
    "ETC": ExtraTreeClassifier()
}

# Fit each model and display output
model_outputs = {}

for name, model in classifiers.items():
    model.fit(X_train, y_train)
    model_outputs[name] = model.__dict__

# For demonstration, let's display the output of the first model
first_model_name = list(model_outputs.keys())[0]
first_model_output = model_outputs[first_model_name]

# Initialize dictionary to store evaluation reports
evaluation_reports = {}

for name, model in classifiers.items():
    # Predicting the Test set results
    y_pred = model.predict(X_test)
    
    # Generating classification report
    report = classification_report(y_test, y_pred, output_dict=True)
    evaluation_reports[name] = report

# For demonstration, let's display the evaluation report of the first model
first_model_evaluation = evaluation_reports[first_model_name]

first_model_name, first_model_output, first_model_evaluation