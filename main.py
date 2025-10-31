from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold, cross_val_predict, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE


#IMPORT DATA

data = pd.read_csv(r"C:\Users\skuba\Downloads\task_data.csv",decimal=",")

print(data.columns.tolist())

X = data[[
    "Heart width","Lung width","CTR - Cardiothoracic Ratio",
    "xx","yy","xy","normalized_diff","Inscribed circle radius",
    "Polygon Area Ratio","Heart perimeter","Heart area " ,"Lung area"
]]
y=data["Cardiomegaly"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  #Splitting data (learning 80%, test 20%)

scaler = StandardScaler()
X_scaled_train = scaler.fit_transform(X_train)
X_scaled_test = scaler.transform(X_test)


#K-Nearest Neighbors (KNN) Classifier


#preliminary diagnosis
#checking the nearest neighbors

pipe_knn = Pipeline(steps=[
    ("scaler", StandardScaler()),
    ("model", KNeighborsClassifier(
        n_neighbors = 3,
        weights='distance',
        metric='manhattan'
    ))
])
pipe_knn.fit(X_train, y_train)


cv_score = np.round(cross_val_score(pipe_knn, X_train, y_train), 2)

#first results
print("Scores of training data cross-validation (each fold):")
list(map(print, cv_score))
print(f"\nCross-validation mean score: {np.mean(cv_score):.3}")
print(f"Standard deviation of CV score: {np.std(cv_score):.3f}")
#it is not bad but it could be better

#now improving acuuracy
#GRID
#finding the best parameter configuration

param_grid = {
    "model__n_neighbors": [3, 5, 7, 9, 11, 15],  # Number of neighbors to consider
    "model__weights": ["uniform", "distance"],   # How neighbors contribute to the prediction
    "model__metric": ["minkowski", "manhattan", "euclidean", "chebyshev"],  # Distance metrics to test
}


rskf = RepeatedStratifiedKFold(
    n_splits=5,
    n_repeats=50,
    random_state=42
)

pipe_knn = Pipeline(steps=[
    ("scaler", StandardScaler()),
    ("model", KNeighborsClassifier())
])

grid_search = GridSearchCV(
    estimator=pipe_knn,
    param_grid=param_grid,
    scoring="accuracy",
    cv=rskf,
    verbose=1,
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

print(f"Best parameters: {grid_search.best_params_}")
print(f"Best accuracy (averaged CV): {grid_search.best_score_:.4f}")

param_grid = {
    "model__n_neighbors": [6, 7, 8],
    "model__weights": ["uniform"],
    "model__metric": ["manhattan"],
}

rskf = RepeatedStratifiedKFold(
    n_splits=5,
    n_repeats=100,
    random_state=None
)

pipe_knn = Pipeline(steps=[
    ("scaler", StandardScaler()),
    ("model", KNeighborsClassifier())
])

grid_search = GridSearchCV(
    estimator=pipe_knn,
    param_grid=param_grid,
    scoring="accuracy",
    cv=rskf,
    verbose=1,
    n_jobs=-1
)

grid_search.fit(X_train, y_train)
print(f"Best parameters: {grid_search.best_params_}")
print(f"Best accuracy (averaged CV): {grid_search.best_score_:.4f}")


best_params = grid_search.best_params_
print(f"Best parameters: {best_params}")
print(f"Best accuracy (averaged CV): {grid_search.best_score_:.4f}")

pipe_knn = Pipeline(steps=[
    ("scaler", StandardScaler()),
    ("model", KNeighborsClassifier(
        n_neighbors = 7,
        weights='uniform',
        metric='manhattan'
    ))
])

pipe_knn.fit(X_train, y_train)

cv_score = np.round(cross_val_score(pipe_knn, X_train, y_train), 2)

#second results after grid
print("Scores of training data cross-validation (each fold):")
list(map(print, cv_score))
print(f"\nCross-validation mean score: {np.mean(cv_score):.3}")
print(f"Standard deviation of CV score: {np.std(cv_score):.3f}")


y_pred = pipe_knn.predict(X_test)#to use the best parametres

#first results on testing data/ KNN statistics
print("\n=== KNN ===")
print("Accuracy on test data:", round(accuracy_score(y_test, y_pred), 3))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
# it is not the best

# The learning data contains more ill patients than healthy ones,so the model tends to predict illness more often because that class dominates.
# SMOTE generates new synthetic samples of the minority class (healthy patients)
# in the training set to make both classes roughly 50/50.
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

print("Przed SMOTE:", np.bincount(y_train))
print("Po SMOTE:", np.bincount(y_resampled))

#training after smoth
#using the best parametres
pipe_knn = Pipeline(steps=[
    ("scaler", StandardScaler()),
    ("model", KNeighborsClassifier(
        n_neighbors = best_params['model__n_neighbors'],
        weights = best_params['model__weights'],
        metric = best_params['model__metric']
    ))
])

pipe_knn.fit(X_resampled, y_resampled)

#second results on training data
y_pred = pipe_knn.predict(X_test)
print("\n=== KNN ===")
print("\n=== TEST RESULTS AFTER SMOTE ===")
print("Accuracy on test data:", round(accuracy_score(y_test, y_pred), 3))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

#result are not satistying so I will use method DECISSION TREE in order to improve my model

#DECISSION TREE

clf_tree = DecisionTreeClassifier(
    max_depth=7,
    criterion='log_loss',
    min_samples_split=7,
    min_samples_leaf=5,
    class_weight=None
)

clf_tree.fit(X_train, y_train)

cv_score = np.round(cross_val_score(clf_tree, X_train, y_train), 2)

pipe_tree = Pipeline(steps=[
    ("scaler", StandardScaler()),
    ("model", DecisionTreeClassifier(
        max_depth=7,
        criterion='entropy',
        min_samples_split=7,
        min_samples_leaf=5,
        class_weight=None
    ))
])

pipe_tree.fit(X_train, y_train)

cv_score = np.round(cross_val_score(pipe_tree, X_train, y_train, cv=5), 2)

print("Scores of training data cross-validation (each fold):")
list(map(print, cv_score))
print(f"\nCross-validation mean score: {np.mean(cv_score):.3}")
print(f"Standard deviation of CV score: {np.std(cv_score):.3f}")

y_pred_tree = pipe_tree.predict(X_test)

print("\n=== DECISION TREE TEST RESULTS ===")
print("Accuracy on test data:", round(accuracy_score(y_test, y_pred_tree), 3))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_tree))
print("\nClassification Report:")
print(classification_report(y_test, y_pred_tree))

#results are not good... model learns well in training, but does not generalize to new data
#RANDOM FOREST

clf_rf = RandomForestClassifier(
    max_depth=6,
    min_samples_split=6,
    n_estimators=125,
    min_samples_leaf=2,
    max_features='sqrt'
)

clf_rf.fit(X_train, y_train)

cv_score = np.round(cross_val_score(clf_rf, X_train, y_train), 2)

print(f"Scores of training data cross-validation (each fold):")
list(map(print, cv_score))
print(f"\nCross-validation mean score: {np.mean(cv_score):.3f}")
print(f"Standard deviation of CV score: {np.std(cv_score):.3f}")

y_pred = clf_rf.predict(X_test)

print("\n=== RANDOM FOREST TEST RESULTS ===")
print("Accuracy on test data:", round(accuracy_score(y_test, y_pred), 3))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
#results are not that bad
#
# now i will combine theese three methods in order to achieve better results

clf_log = Pipeline(steps=[
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(max_iter=1000, class_weight='balanced'))
])

clf_log.fit(X_resampled, y_resampled)
y_pred_log = clf_log.predict(X_test)

print("\n=== LOGISTIC REGRESSION TEST RESULTS ===")
print("Accuracy on test data:", round(accuracy_score(y_test, y_pred_log), 3))
print(confusion_matrix(y_test, y_pred_log))
print(classification_report(y_test, y_pred_log))
#much better, this method improved results a lot

#last try of improving result
#SVC
pipe_svc = Pipeline(steps=[
    ("scaler", StandardScaler()),
    ("model", SVC(                  # Support Vector Classifier
        kernel="rbf",               # RBF kernel captures non-linear decision boundaries
        C=3,                        # Regularization strength (higher = tighter fit to training data)
        gamma="scale",              # Kernel width; 'scale' adapts to data variance
        class_weight=None           # Treat classes equally (no re-weighting)
    ))
])

pipe_svc.fit(X_train, y_train)

cv_score = np.round(cross_val_score(pipe_svc, X_train, y_train), 2)

param_grid_svc = {
    "model__C": [0.1, 1, 3, 10, 30],
    "model__gamma": ["scale", 0.01, 0.001, 0.0001],
    "model__kernel": ["rbf", "poly", "sigmoid"]
}

grid_search_svc = GridSearchCV(
    estimator=pipe_svc,
    param_grid=param_grid_svc,
    scoring="accuracy",
    cv=5,
    verbose=1,
    n_jobs=-1
)

grid_search_svc.fit(X_train, y_train)

print(f"Best parameters: {grid_search_svc.best_params_}")
print(f"Best cross-validation accuracy: {grid_search_svc.best_score_:.4f}")


print("Scores of training data cross-validation (each fold):")
list(map(print, cv_score))
print(f"\nCross-validation mean score: {cv_score.mean():.3f}")
print(f"Standard deviation of CV score: {cv_score.std():.3f}")

best_svc = grid_search_svc.best_estimator_

y_pred = best_svc.predict(X_test)

print("\n=== SVM TEST RESULTS ===")
print("Accuracy on test data:", round(accuracy_score(y_test, y_pred), 3))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

knn_best = Pipeline(steps=[
    ("scaler", StandardScaler()),
    ("model", KNeighborsClassifier(
        n_neighbors=6,
        weights='uniform',
        metric='manhattan'
    ))
])

tree_best = DecisionTreeClassifier(
    max_depth=7,
    criterion='log_loss',
    min_samples_split=7,
    min_samples_leaf=5
)

rf_best = RandomForestClassifier(
    max_depth=6,
    min_samples_split=6,
    n_estimators=125,
    min_samples_leaf=2,
    max_features='sqrt'
)

log_reg = Pipeline(steps=[
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(
        C=1.0,
        solver='lbfgs',
        max_iter=1000
    ))
])

svc_best = Pipeline(steps=[
    ("scaler", StandardScaler()),
    ("model", SVC(
        C=3,
        kernel='rbf',
        gamma='scale',
        probability=True
    ))
])


models = {
    "KNN": Pipeline(steps=[("scaler", StandardScaler()),
                           ("model", KNeighborsClassifier(n_neighbors=6, weights='uniform', metric='manhattan'))]),
    "Decision Tree": DecisionTreeClassifier(max_depth=7, criterion='log_loss', min_samples_split=7, min_samples_leaf=5),
    "Random Forest": RandomForestClassifier(max_depth=6, min_samples_split=6, n_estimators=125, min_samples_leaf=2,
                                            max_features='sqrt'),
    "Logistic Regression": Pipeline(steps=[("scaler", StandardScaler()),
                                           ("model", LogisticRegression(max_iter=1000, class_weight='balanced'))]),
    "SVC": Pipeline(steps=[("scaler", StandardScaler()),
                           ("model", SVC(C=3, kernel='rbf', gamma='scale', probability=True))])
}


results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    results[name] = acc

    print("\n" + "=" * 40)
    print(f"=== {name} TEST RESULTS ===")
    print("=" * 40)
    print("Accuracy on test data:", round(acc, 3))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

#final results
print("=== RESULTS COMPARISON ===")
for name, acc in results.items():
    print(f"{name}: Accuracy = {acc:.3f}")




