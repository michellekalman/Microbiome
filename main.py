import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from data_cleaning import microbiome, metadata, test_microbiome, test_metadata
from data_exploration import important_bacteria, bray_curtis_dissimilarity

from model import *

train_sample = metadata['sample']
test_sample = microbiome['sample']

X_train = metadata.drop(['sample', 'collection_date','time_diff', "baboon_id"], axis=1)
y_train = microbiome.drop(['sample'], axis=1)

X_test = test_metadata.drop(['collection_date','time_diff', "baboon_id"], axis=1)
y_test = test_microbiome

X_test_check = X_test[X_test['sample'].isin(y_test['sample'])]
X_test_predict = X_test[~X_test['sample'].isin(y_test['sample'])]

y_test = y_test.drop(['sample'], axis=1)
X_test_check = X_test_check.drop(['sample'], axis=1)
X_test_predict = X_test_predict.drop(['sample'], axis=1)
X_test = X_test.drop(['sample'], axis=1)

X_train = X_train
X_test_check = X_test_check
y_train = y_train
X_test_predict = X_test_predict
y_test = y_test
X_test = X_test
print("X_train:\n", X_train.columns)
print("y_train:\n", y_train.columns)
print("X_test_check:\n", X_test_check.columns)
print("X_test_predict:\n", X_test_predict.columns)
print("y_test:\n", y_test.columns)
print("X_test:\n", X_test.columns)

if __name__ == "__main__":
    train_sample.to_csv("sample.csv", index=False)
    print("\nhi\n")
    rfr = RandomForestRegressor(random_state=42)
    gbr = GradientBoostingRegressor(random_state=42)

    selected_features_dict_rf, y_pred_rf = RSECV_for_important_bacteria(2, rfr, X_train, y_train, X_test, important_bacteria)
    selected_features_dict_gb, y_pred_gb = RSECV_for_important_bacteria(2, gbr, X_train, y_train, X_test, important_bacteria)

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
        output.to_csv("predictions_{model}_normal.csv".format(model=mod), index=False)


    # print(y_pred_gb)
    print("========================== finished with stage 1 ===================================")
    X_test_new, y_test_new, X_train_new, y_train_new = merge_important_bacteria_with_metadata(X_test, y_pred_gb, y_test, X_train, y_train)
    print("\n",X_test_new,"\n", y_test_new,"\n", X_train_new,"\n", y_train_new)
    rf_reg = RandomForestRegressor(random_state=42)
    gb_reg = GradientBoostingRegressor(random_state=42)
    selected_features_dict2_rf = RSECV_for_unimportant_bacteria_selected_features_meta(2, rf_reg, X_train_new, y_train_new, X_test_new, selected_features_dict_rf)
    selected_features_dict2_gb = RSECV_for_unimportant_bacteria_selected_features_meta(2, gb_reg, X_train_new, y_train_new, X_test_new, selected_features_dict_gb)
    
    print("========================== finished with stage 2 ===================================")
    y_pred2_rf = predict_unimportant_baecteria(rf_reg, X_train_new, y_train_new, X_test_new, selected_features_dict2_rf)
    y_pred2_gb = predict_unimportant_baecteria(gb_reg, X_train_new, y_train_new, X_test_new, selected_features_dict2_rf)

    print("========================== finished with stage 3 ===================================")
    output2_rf = pd.DataFrame(y_pred2_rf)
    output2_rf.to_csv("predictions2_rf.csv", index=False)
    output2_gb = pd.DataFrame(y_pred2_gb)
    output2_gb.to_csv("predictions2_gb.csv", index=False)
    '''
    scores_gb = evaluate_model(y_test, y_pred2_gb)
    socres_dissimilarity_gb = bray_curtis_dissimilarity(y_pred2_gb, y_test)
    scores_rf = evaluate_model(y_test, y_pred2_rf,)
    socres_dissimilarity_rf = bray_curtis_dissimilarity(y_pred2_rf, y_test)

    scores_gb_norm = evaluate_model(y_test, y_pred_gb)
    scores_rf_norm = evaluate_model(y_test, y_pred_rf)

    print("from random forest:")
    print("based on new data:")
    print("bray curtis:")
    print(scores_rf)
    print("bray curtis dissimilarity - manually calculated:")
    print(socres_dissimilarity_rf)
    print("normal random forest:")
    print(scores_rf_norm)

    print("from gradient boost:")
    print("based on new data:")
    print("bray curtis:")
    print(scores_gb)
    print("bray curtis dissimilarity - manually calculated:")
    print(socres_dissimilarity_gb)
    print("normal random forest:")
    print(scores_gb_norm)
    # we see slightly better results on our model!'''