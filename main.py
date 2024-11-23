import pandas as pd
import numpy as np
from scipy.spatial.distance import braycurtis
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from data_exploration import important_bacteria
from data_cleaning import microbiome, test_metadata, train_metadata
from data_exploration import bray_curtis_dissimilarity, microbiome_data, important_bacteria

from model import *

X_train = train_metadata.drop(['sample', 'collection_date','time_diff', "baboon_id"], axis=1)
y_train = microbiome.drop(['sample'], axis=1)
X_test = test_metadata.drop(['sample', 'collection_date','time_diff', "baboon_id"], axis=1)
y_test = np.zeros((len(X_test), len(y_train.columns)))
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
        y_pred={}
        if mod == 'rf':
            mod = rf_reg
            scores_normal_model = scores_normal_rf
        else:
            mod = gb_reg
            scores_normal_model = scores_normal_gb
        for col in important_bacteria:
            mod.fit(X_train, y_train[col])
            y_pred[col] = mod.predict(X_test)
#            scores_normal_model[col] = braycurtis(y_pred[col], y_test[col])
        output = pd.DataFrame(y_pred)
        output.to_csv("predictions{model}.csv".format(model=mod), index=False)

    # Compare
    # print("from random forest:")
    # print("normal:")
    # print(scores_normal_rf)
    # print("feature selection:")
    # print(rf_by_important_scores)
    # print("from gradient boost:")
    # print("normal:")
    # print(scores_normal_gb)
    # print("feature selection:")
    # print(gb_by_important_scores)


    # print(y_pred_gb)

    X_test_new, y_test_new, X_train_new, y_train_new = merge_important_bacteria_with_metadata(X_test, y_pred_gb, y_test, X_train, y_train)

    # Compare to same model based on all features
    
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