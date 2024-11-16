import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
import seaborn as sns
from scipy.spatial.distance import braycurtis
from sklearn.feature_selection import RFE
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor

from data_cleaning import metadata, pd, combined_df, features, microbiome
from data_exploration import bray_curtis_dissimilarity, microbiome_data, important_bacteria

# Split the data into training and testing sets
X = metadata.drop(['sample', 'collection_date', 'time_diff', "baboon_id"], axis=1)
y = microbiome.drop(['sample'], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Initialize and train the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

embeded_feature_importance ={}
for i in range(len(model.feature_importances_)):
    embeded_feature_importance[X.columns[i]] = model.feature_importances_[i]

print(embeded_feature_importance)

plt.figure(figsize=(10, 6))
sns.barplot(embeded_feature_importance)
plt.title('Feature importance - metadata', fontsize=16)
plt.xlabel('Feature', fontsize=12)
plt.ylabel('importance', fontsize=12)
plt.xticks(rotation=90)
plt.grid(True)
plt.show()

# Calculate the Bray-Curtis dissimilarity
dissimilarity_score = bray_curtis_dissimilarity(y_test.values, y_pred)

print(dissimilarity_score)


def brey_curtis_distance(arr1, arr2):
    dists = [float('inf') for i in range(len(arr1))]
    for i in range(len(arr1)):
        dists[i] = braycurtis(arr1[i,:], arr2[i,:])
    return dists
brey_curtis_dist = brey_curtis_distance(y_test.values, y_pred)
print(brey_curtis_dist)

Y_df = pd.DataFrame(y_pred)
print(model.feature_importances_)

metadata['day_of_month'] = metadata['collection_date'].dt.day
metadata['week_of_year'] = metadata['collection_date'].dt.isocalendar().week
metadata['year'] = metadata['collection_date'].dt.year




model = MultiOutputRegressor(RandomForestRegressor())
model = model.fit(X_train, y_train)

rfe_dict = {}
for bacteria_idx in range(len(y.columns)):
    if bacteria_idx != 5:
        continue
    print(f"Bacteria: {y.columns[bacteria_idx]}")
    est_num = bacteria_idx
    est = model.estimators_[est_num]
    feature_importances = pd.DataFrame(est.feature_importances_, columns=['importance']).sort_values('importance')
    feature_importances.plot(kind = 'barh', figsize=(10, 6), title=y.columns[bacteria_idx])
    plt.show()


    selector = RFE(est, step=1)
    selector = selector.fit(X_train, y_train.iloc[:, bacteria_idx])
    print(selector.support_)
    rfe_dict[y.columns[bacteria_idx]] = selector.support_

for bact in rfe_dict:
    mask = rfe_dict.get(bact)
    rfe_dict[bact] = X.columns[mask]

print(rfe_dict)

sumup_dict = {}
for bact in rfe_dict.values():
    for feature in bact:
        sumup_dict[feature] = sumup_dict.get(feature, 0) + 1

print(sumup_dict)

plt.figure(figsize=(10, 6))
sns.barplot(x=list(sumup_dict.values()), y=list(sumup_dict.keys()))

from sklearn.feature_selection import RFECV
from sklearn.model_selection import cross_val_score, KFold

X = metadata.drop(['sample', 'collection_date', 'time_diff', "baboon_id"], axis=1)
y = microbiome.drop(['sample'], axis=1)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=0)
kf = KFold(n_splits=2, shuffle=True, random_state=0)  # 5-fold cross-validation

# Dictionary to store selected features for each bacteria
selected_features_dict = {}
y_pred = {}
# Loop through each important bacteria and perform RFECV
for bacteria in important_bacteria:
    y_train_bacteria = y_train[bacteria]  # The target is the bacterial percentage for this bacteria

    # Initialize the model (RandomForest for regression)
    model = RandomForestRegressor()

    # Initialize RFECV with the model and cross-validation
    rfecv = RFECV(estimator=model, step=1, cv=kf, scoring='neg_mean_squared_error')  # Adjust 'scoring' if needed

    # Fit RFECV to the data
    rfecv.fit(X_train, y_train_bacteria)

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
    plt.ylabel("Cross-Validated MSE")
    plt.title("RFECV Feature Selection with Cross-Validation")
    plt.show()

    # Get the optimal number of features and the selected features
    optimal_num_features = rfecv.n_features_
    selected_features = X_train.columns[rfecv.support_]

    print(f"Optimal number of features selected: {optimal_num_features}")
    print(f"Selected features for common bacteria: {selected_features}")

    model.fit(X_train[selected_features], y_train_bacteria)
    y_pred_bacteria = model.predict(X_test[selected_features])
    y_pred[bacteria] = y_pred_bacteria

# Convert the selected features dictionary to a DataFrame for easier viewing
selected_features_df = pd.DataFrame.from_dict(selected_features_dict, orient='index').transpose()
print(selected_features_df)


# This code is for the borua algorithm we tried to implement on our data,
# we didn't get the results we were looking for even though we tried different alpha values :(

X = metadata.drop(['sample', 'collection_date', 'time_diff', "baboon_id"], axis=1)
y = microbiome.drop(['sample'], axis=1)

# Compute the correlation matrix
correlation_matrix = X.corr().abs()

# Identify features that are highly correlated (threshold can be adjusted)
threshold = 0.85
upper_triangle = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))

# Find features to drop (those with correlation > threshold)
to_drop = [column for column in upper_triangle.columns if any(upper_triangle[column] > threshold)]
print("Highly correlated features to drop:", to_drop)

# Drop the highly correlated features from X
X_uncorrelated = X.drop(columns=to_drop)

from boruta import BorutaPy
from sklearn.ensemble import RandomForestRegressor

# y_common is the bacterial percentage of the most common bacteria (replace 'most_common_bacteria' with actual column)
y_common = [microbiome_data[i] for i in important_bacteria] # E.g., 'Bacteria_A'

# Initialize a RandomForestRegressor
model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)

# Initialize Boruta with the RandomForest model
boruta_selector = BorutaPy(estimator=model, n_estimators='auto', random_state=42,  alpha=0.4)

for i in range(len(important_bacteria)):
  # Fit Boruta to the data (X_uncorrelated) and y_common
  boruta_selector.fit(X_uncorrelated.values, y_common[i])

  # Get the selected features after running Boruta
  selected_features = X_uncorrelated.columns[boruta_selector.support_].tolist()
  print("Selected features from Boruta:", selected_features)

  # Create a reduced dataset with the selected features
  X_selected_boruta = X_uncorrelated[selected_features]