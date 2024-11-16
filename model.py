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


def RSECV_for_important_bacteria(k_fold, model, X_train, y_train, X_test, important_bacteria):
    kf = KFold(n_splits=k_fold, shuffle=True, random_state=42)  # cross-validation

    # Dictionary to store selected features for each bacteria
    selected_features_dict = {}
    y_pred = {}

    # Loop through each important bacteria and perform RFECV
    for bacteria in important_bacteria:
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
        plt.show()

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
        plt.show()

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
        plt.show()

        # Get the optimal number of features and the selected features
        optimal_num_features = rfecv.n_features_
        selected_features = X_train_new.columns[rfecv.support_]

        print(f"Optimal number of features selected: {optimal_num_features}")
        print(f"Selected features for common bacteria: {selected_features}")

    return selected_features_dict

if __name__ == "__main__":
    rfr = RandomForestRegressor(random_state=42)
    gbr = GradientBoostingRegressor(random_state=42)
    selected_features_dict_rf, y_pred_rf = RSECV_for_important_bacteria(2, rfr, X_train, y_train, X_test, important_bacteria)
    print(selected_features_dict_rf, y_pred_rf)

    selected_features_dict_gb, y_pred_gb = RSECV_for_important_bacteria(2, gbr, X_train, y_train, X_test, important_bacteria)

    # Peak to  see the important features
    print("from random forest:")
    print(selected_features_dict_rf)
    print("from gradient boost:")
    print(selected_features_dict_gb)

    # Lets check our predictions for the important bacteria based on their important features
    rf_by_important_scores = evaluate_model(y_test, y_pred_rf)
    gb_by_important_scores = evaluate_model(y_test, y_pred_gb)

    # Compare to same model based on all features
    from scipy.spatial.distance import braycurtis
    rf_reg = RandomForestRegressor(random_state=42)
    gb_reg = GradientBoostingRegressor(random_state=42)
    scores_normal_rf = {}
    scores_normal_gb = {}
    for mod in ['rf', 'gb']:
        if mod == 'rf':
            mod = rf_reg
            scores_normal_model = scores_normal_rf
        else:
            mod = gb_reg
            scores_normal_model = scores_normal_gb
        for col in important_bacteria:
            mod.fit(X_train, y_train[col])
            y_pred = mod.predict(X_test)
            scores_normal_model[col] = braycurtis(y_pred, y_test[col])

    # Compare
    print("from random forest:")
    print("normal:")
    print(scores_normal_rf)
    print("feature selection:")
    print(rf_by_important_scores)
    print("from gradient boost:")
    print("normal:")
    print(scores_normal_gb)
    print("feature selection:")
    print(gb_by_important_scores)


    print(y_pred_gb)

    X_test_new, y_test_new, X_train_new, y_train_new = merge_important_bacteria_with_metadata(X_test, y_pred_gb, y_test, X_train, y_train)

    # Compare to same model based on all features
    from scipy.spatial.distance import braycurtis
    rf_reg = RandomForestRegressor(random_state=42)
    gb_reg = GradientBoostingRegressor(random_state=42)
    scores_normal_rf_new = {}
    scores_normal_gb_new = {}
    for mod in ['rf', 'gb']:
        if mod == 'rf':
            mod = rf_reg
            scores_normal_model = scores_normal_rf_new
        else:
            mod = gb_reg
            scores_normal_model = scores_normal_gb_new
        for col in important_bacteria:
            mod.fit(X_train_new, y_train_new[col])
            y_pred = mod.predict(X_test_new)
            scores_normal_model[col] = braycurtis(y_pred, y_test_new[col])

    print("from random forest:")
    print("based on new data:")
    print(scores_normal_rf_new)
    print("normal:")
    print(scores_normal_rf)
    print("from gradient boost:")
    print("our data:")
    print(scores_normal_gb_new)
    print("normal:")
    print(scores_normal_gb)

    # we see slightly better results on our model!