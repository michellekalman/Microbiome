import pandas as pd
import json 
from matplotlib import pyplot as plt
from scipy.spatial.distance import braycurtis
from sklearn.feature_selection import RFECV
from sklearn.model_selection import KFold

'''
in:
k-fold = number of splits for cv
model = estimator with feature importance (embeded model)
X_train = metadata
y_train = microbiome data
important_bacteria = list of bacteria to be used

out:
dict of important bacteria and their selected features
dict of prediction for important bacteria
'''
def RSECV_for_important_bacteria(k_fold, model, X_train, y_train, X_test, important_bacteria):
    print("$$$$$$$$$$$$$$$$$$$$$$$$$$ RSECV_for_important_bacteria")
    kf = KFold(n_splits=k_fold, shuffle=True, random_state=42)  # cross-validation

    # Dictionary to store selected features for each bacteria
    selected_features_dict = {}
    y_pred = {}

    # Loop through each important bacteria and perform RFECV
    for bacteria in important_bacteria:
        y = y_train[bacteria]  # The target is the bacterial percentage for this bacteria

        rfecv = RFECV(estimator=model, step=1, cv=kf, scoring='neg_mean_squared_error')
        rfecv.fit(X_train, y)

        # Store the selected features for this bacteria
        selected_features = X_train.columns[rfecv.support_]
        selected_features_dict[bacteria] = selected_features

        # Use the new attribute 'cv_results_' (mean_test_score stores cross-validated scores)
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(rfecv.cv_results_['mean_test_score']) + 1), rfecv.cv_results_['mean_test_score'])
        plt.xlabel("Number of Features Selected")
        plt.ylabel("Cross-Validated dists (neg)")
        plt.title("RFECV Feature Selection with Cross-Validation")
        plt.savefig("RFECV_important_bacteria_{bact}".format(bact=bacteria))

        model.fit(X_train[selected_features], y)
        y_pred_bacteria = model.predict(X_test[selected_features])
        y_pred[bacteria] = y_pred_bacteria

    print("/\/\/\/\///\/\/\/\/\savinf dict for debug/\/\//\/\/\/\/\/\/\/\/")
    with open('selected_features1{mod}.txt'.format(mod=model), 'w') as f:  
        for key, value in selected_features_dict.items():  
            f.write('%s:%s\n' % (key, value))
        f.close

    return selected_features_dict, y_pred

'''
in:
y_test = test microbiome data
y_pred = dict of bacteria and their predicted values

out:
dict of bacteria and their dists
'''

def evaluate_model(y_test, y_pred):
    dists = {}
    for bacteria in y_pred.keys():
        dists[bacteria] = braycurtis(y_test[bacteria], y_pred[bacteria])
    return dists

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
    print("!++++++++++++++++!!!!!!!!!!!!!!!!!!!!merge")
    important_bact = list(y_pred.keys())

    y_pred = pd.DataFrame(y_pred)

    y_pred.reset_index(drop=True, inplace=True)
    X_test.reset_index(drop=True, inplace=True)
    y_test.reset_index(drop=True, inplace=True)
    y_train.reset_index(drop=True, inplace=True)
    X_train.reset_index(drop=True, inplace=True)

    X_test_new = pd.concat([X_test, y_pred], axis=1)

    y_test_new = y_test.drop(important_bact, axis=1)

    X_train_new = pd.concat([X_train, y_train[important_bact]], axis=1)

    y_train_new = y_train.drop(important_bact, axis=1)

    tmp_dict = {'X_test_new':X_test_new, 'y_test_new':y_test_new, 'X_train_new':X_train_new, 'y_train_new':y_train_new}
    for lable, data_set in tmp_dict.items():    
        output = pd.DataFrame(data_set)
        output.to_csv("merge_func_{name}.csv".format(name=lable), index=False)

    return X_test_new, y_test_new, X_train_new, y_train_new




'''
filter selected features that we got for the important bacteria, add the bacteria and use only these to predict microbiome
'''

def RSECV_for_unimportant_bacteria_selected_features_meta(k_fold, model, X_train_new, y_train_new, X_test_new, selected_features_dict):
    print("~~~~~~~~~~^^^^^^^^^^ RSECV_for_unimportant_bacteria_selected_features_meta")
    kf = KFold(n_splits=k_fold, shuffle=True, random_state=42)  # cross-validation

    selected_features_dict = {}

    for bacteria in list(y_train_new.columns):
        y = y_train_new[bacteria]  # The target is the bacterial percentage for this bacteria

        rfecv = RFECV(estimator=model, step=1, cv=kf, scoring='neg_mean_squared_error')  # Adjust 'scoring' if needed

        # Fit RFECV to the data
        rfecv.fit(X_train_new, y)

        # Store the selected features for this bacteria
        selected_features = X_train_new.columns[rfecv.support_]
        selected_features_dict[bacteria] = selected_features


        # Use the new attribute 'cv_results_' (mean_test_score stores cross-validated scores)
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(rfecv.cv_results_['mean_test_score']) + 1), rfecv.cv_results_['mean_test_score'])
        plt.xlabel("Number of Features Selected")
        plt.ylabel("Cross-Validated dists (neg)")
        plt.title("RFECV Feature Selection with Cross-Validation")
        plt.savefig("rsev_unimportant_selected_{bact}".format(bact=bacteria))


    with open('selected_features2{mod}.txt'.format(mod=model), 'w') as f:  
        for key, value in selected_features_dict.items():  
            f.write('%s:%s\n' % (key, value))
    return selected_features_dict


def predict_unimportant_baecteria(model, X_train, y_train, X_test, selected_features_dict):
    print("===========================in predict func===========")
    selected_features = []
    for bacteria in selected_features_dict.values():
        selected_features.extend(bacteria)
    
    selected_features = list(set(selected_features))
    X_train_new = X_train[selected_features]
    X_test_new = X_test[selected_features]
    
    y_pred = {}
    for bacteria in y_train.columns:
        y = y_train[bacteria]
        model.fit(X_train_new[selected_features], y)
        y_pred_bacteria = model.predict(X_test_new[selected_features])
        y_pred[bacteria] = y_pred_bacteria
    
    return y_pred

    
