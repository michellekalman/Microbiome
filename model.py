'''
in:
k-fold = number of splits for cv
model = estimator with feature importance (embeded model)
X_train = metadata
y_train = microbiome data
important_bacteria = list of bacteria to be used

out:
dict of bacteria and their selected features
dict of prediction for important bacteria
'''
from matplotlib import pyplot as plt
from scipy.spatial.distance import braycurtis
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.feature_selection import RFECV
from sklearn.model_selection import KFold

from data_exploration import important_bacteria
from model_exploration import X_train, y_train, X_test, y_test

X_train_tmp = X_train
y_train_tmp = y_train
X_test_tmp = X_test
y_test_tmp = y_test


def RSECV_for_important_bacteria(k_fold, model, X_train, y_train, X_test, important_bacteria):
    kf = KFold(n_splits=k_fold, shuffle=True, random_state=42)  # cross-validation

    # Dictionary to store selected features for each bacteria
    selected_features_dict = {}
    y_pred = {}

    # Loop through each important bacteria and perform RFECV
    for bacteria in important_bacteria[0]:
        y = y_train[bacteria]  # The target is the bacterial percentage for this bacteria


        # Initialize RFECV with the model and cross-validation
        rfecv = RFECV(estimator=model, step=1, cv=kf, scoring='neg_mean_squared_error')

        # Fit RFECV to the data
        rfecv.fit(X_train, y)

        # Store the selected features for this bacteria
        selected_features = X_train.columns[rfecv.support_]
        selected_features_dict[bacteria] = selected_features

        # Print the results for this bacteria
        print(f"Bacteria: {bacteria}")
        print(f"Optimal number of features: {rfecv.n_features_}")
        print(f"Selected features: {selected_features}")

        # Use the new attribute 'cv_results_' (mean_test_score stores cross-validated scores)
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(rfecv.cv_results_['mean_test_score']) + 1), rfecv.cv_results_['mean_test_score'])
        plt.xlabel("Number of Features Selected")
        plt.ylabel("Cross-Validated MSE (neg)")
        plt.title("RFECV Feature Selection with Cross-Validation")
        plt.savefig("RFECV_important_bacteria_{bact}".format(bact=bacteria))

        # Get the optimal number of features and the selected features
        optimal_num_features = rfecv.n_features_
        selected_features = X_train.columns[rfecv.support_]

        print(f"Optimal number of features selected: {optimal_num_features}")
        print(f"Selected features for common bacteria: {selected_features}")

        model.fit(X_train[selected_features], y)
        y_pred_bacteria = model.predict(X_test[selected_features])
        y_pred[bacteria] = y_pred_bacteria

    return selected_features_dict, y_pred

'''
in:
y_test = test microbiome data
y_pred = dict of bacteria and their predicted values

out:
dict of bacteria and their mse
'''

def evaluate_model(y_test, y_pred):
    mse = {}
    for bacteria, y_pred_bacteria in y_pred.items():
        mse[bacteria] = braycurtis(y_test[bacteria], y_pred_bacteria)
    return mse

'''
in:
X_test = test metadata
y_pred = dict of bacteria and their predicted values
y_test = test microbiome data
X_train = train metadata
y_train = train microbiome data

out:
X_test_new = test metadata with predicted values for important bacteria
y_test_new = test microbiome data without important bacteria
X_train_new = train metadata with predicted values for important bacteria
y_train_new = train microbiome data without important bacteria
'''


def merge_important_bacteria_with_metadata(X_test, y_pred, y_test, X_train, y_train):
    X_test_new = X_test.copy()
    for bacteria, y_pred_bacteria in y_pred.items():
        X_test_new[bacteria] = y_pred_bacteria

    y_test_new = y_test.copy()
    y_test_new.drop([k for k in y_pred.keys()], axis=1)

    X_train_new = X_train.copy()
    for bacteria in y_pred.keys():
        X_train_new[bacteria] = y_train[bacteria].values

    y_train_new = y_train.copy()
    y_train_new.drop([k for k in y_pred.keys()], axis=1)

    return X_test_new, y_test_new, X_train_new, y_train_new

'''
filter selected features that we got for the important bacteria, add the bacteria and use only these to predict microbiome
'''

def RSECV_for_unimportant_bacteria_selected_features_meta(k_fold, model, X_train_new, y_train_new, X_test_new, selected_features_dict):
    kf = KFold(n_splits=k_fold, shuffle=True, random_state=42)  # cross-validation
    selected_features = [b for b in selected_features_dict.keys()]
    for bacteria in selected_features_dict.values():
        selected_features.extend(bacteria)
    selected_features = list(set(selected_features))
    X_train_new = X_train_new[selected_features]

    # Dictionary to store selected features for each bacteria
    selected_features_dict = {}
    y_pred = {}

    for bacteria in list(y_train_new.columns[0]):
        y = y_train_new[bacteria]  # The target is the bacterial percentage for this bacteria

        # Initialize RFECV with the model and cross-validation
        rfecv = RFECV(estimator=model, step=1, cv=kf, scoring='neg_mean_squared_error')  # Adjust 'scoring' if needed

        # Fit RFECV to the data
        rfecv.fit(X_train_new, y)

        # Store the selected features for this bacteria
        selected_features = X_train_new.columns[rfecv.support_]
        selected_features_dict[bacteria] = selected_features

        # Print the results for this bacteria
        print(f"Bacteria: {bacteria}")
        print(f"Optimal number of features: {rfecv.n_features_}")
        print(f"Selected features: {selected_features}")

        # Use the new attribute 'cv_results_' (mean_test_score stores cross-validated scores)
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(rfecv.cv_results_['mean_test_score']) + 1), rfecv.cv_results_['mean_test_score'])
        plt.xlabel("Number of Features Selected")
        plt.ylabel("Cross-Validated MSE (neg)")
        plt.title("RFECV Feature Selection with Cross-Validation")
        plt.savefig("rsev_unimportant_selected_{bact}").format(bact=bacteria)

        # Get the optimal number of features and the selected features
        optimal_num_features = rfecv.n_features_
        selected_features = X_train_new.columns[rfecv.support_]

        print(f"Optimal number of features selected: {optimal_num_features}")
        print(f"Selected features for common bacteria: {selected_features}")

        model.fit(X_train_new, y)
        y_pred_bacteria = model.predict(X_test_new)
        y_pred[bacteria] = y_pred_bacteria

    return selected_features_dict, y_pred


def RSECV_for_unimportant_bacteria_full_metadata(k_fold, model, X_train_new, y_train_new):
    kf = KFold(n_splits=k_fold, shuffle=True, random_state=42)  # cross-validation

    selected_features_dict = {}

    for bacteria in list(y_train_new.columns[0]):
        y = y_train_new[bacteria]  # The target is the bacterial percentage for this bacteria

        # Initialize RFECV with the model and cross-validation
        rfecv = RFECV(estimator=model, step=1, cv=kf, scoring='neg_mean_squared_error')  # Adjust 'scoring' if needed

        # Fit RFECV to the data
        rfecv.fit(X_train_new, y)

        # Store the selected features for this bacteria
        selected_features = X_train_new.columns[rfecv.support_]
        selected_features_dict[bacteria] = selected_features

        # Print the results for this bacteria
        print(f"Bacteria: {bacteria}")
        print(f"Optimal number of features: {rfecv.n_features_}")
        print(f"Selected features: {selected_features}")

        # Use the new attribute 'cv_results_' (mean_test_score stores cross-validated scores)
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(rfecv.cv_results_['mean_test_score']) + 1), rfecv.cv_results_['mean_test_score'])
        plt.xlabel("Number of Features Selected")
        plt.ylabel("Cross-Validated MSE (neg)")
        plt.title("RFECV Feature Selection with Cross-Validation")
        plt.savefig("RFCV_unimpportant_bact_{bact}".format(bact=bacteria))

        # Get the optimal number of features and the selected features
        optimal_num_features = rfecv.n_features_
        selected_features = X_train_new.columns[rfecv.support_]

        print(f"Optimal number of features selected: {optimal_num_features}")
        print(f"Selected features for common bacteria: {selected_features}")

    return selected_features_dict
