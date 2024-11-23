import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from matplotlib import pyplot as plt
from scipy.spatial.distance import pdist, squareform
from sklearn.manifold import MDS
from scipy.spatial.distance import braycurtis

from data_cleaning import metadata,test_metadata, train_metadata, combined_df, features, microbiome

metadata = train_metadata
# Q1 - how many samples from each subject
sample_cnt_per_baboon = metadata.groupby(["baboon_id"]).count()
sample_counts = pd.DataFrame(sample_cnt_per_baboon['sample'])
sample_counts.head(10)

# Q1 - time differences
# Calculate the time difference between samples
metadata['time_diff'] = metadata.groupby('baboon_id')['collection_date'].diff()

# Display the result
print(metadata[['baboon_id', 'collection_date', 'time_diff']])
metadata['time_diff'] = metadata.groupby('baboon_id')['collection_date'].diff()
time_diff = metadata[['baboon_id', 'collection_date', 'time_diff']]
print(time_diff.loc[time_diff['baboon_id']==4])

# Filter out NaT values from time_diff
time_diff_days = metadata['time_diff'].dropna().dt.days
combined_df['time_diff'] = time_diff_days

plt.figure(figsize=(10, 6))
plt.hist(time_diff_days, bins=80, range=(0, 200), edgecolor='black')
plt.title('Distribution of Time Differences Between Samples')
plt.xlabel('Time Difference (days)')
plt.ylabel('Frequency')
plt.savefig('Distribution_Time_Differences_Between_Samples.pdf')


# Filter out NaT values
meta_filtered = metadata.dropna(subset=['time_diff'])

# Convert time_diff to days for the box plot
meta_filtered['time_diff_days'] = meta_filtered['time_diff'].dt.days

plt.figure(figsize=(14, 8))
sns.boxplot(x='baboon_id', y='time_diff_days', data=meta_filtered)
plt.title('Time Differences Between Samples for Each Baboon')
plt.xlabel('Baboon ID')
plt.ylabel('Time Difference (days)')
plt.ylim(0, 600)
plt.xticks(rotation=90, fontsize=8)
plt.savefig('Time_Differences_Between_Samples_for_Each_Baboon.pdf')

# The visualization of this data was not very helpfull for our understanding because the largs number of the baboons.
# What helped us understand this point was the below statistics.
# Filter out NaT values
meta_filtered_filtered = metadata.dropna(subset=['time_diff'])

# Convert time_diff to days for the box plot
meta_filtered['time_diff_days'] = meta_filtered['time_diff'].dt.days

plt.figure(figsize=(14, 8))
sns.boxplot(x='baboon_id', y='time_diff_days', data=meta_filtered)
plt.title('Time Differences Between Samples for Each Baboon')
plt.xlabel('Baboon ID')
plt.ylabel('Time Difference (days)')
plt.ylim(0, 150)
plt.xticks(rotation=90)
plt.savefig('time_diff_per_baboon.pdf')

filtered_df = time_diff.loc[metadata['time_diff'].dt.days > 1000]

print(filtered_df)

# Calculate the correlation matrix
correlation_matrix = combined_df[features].corr()

# Display the correlation matrix

plt.figure(figsize=(15, 10))
sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', center=0)
plt.title('Correlation Matrix Heatmap')
plt.savefig('Correlation_Matrix_Heatmap.pdf')

# We observe that there is no obviouse correlation between features.

# Calculate Bray-Curtis distance matrix
sorted_microbiome = pd.DataFrame(combined_df[microbiome.columns])
bray_curtis_distances = pdist(sorted_microbiome.iloc[:, 1:], metric='braycurtis')
bray_curtis_matrix = squareform(bray_curtis_distances)

# Display the distance matrix
bray_curtis_df = pd.DataFrame(bray_curtis_matrix, index=sorted_microbiome.index, columns=sorted_microbiome.index)
print(bray_curtis_df)

samples_per_baboon2 = combined_df.groupby('baboon_id')['sample'].apply(list).reset_index()

for baboon in samples_per_baboon2['baboon_id']:
    print('bray curtis subsequental dists for baboon: ', baboon)
    samples = list(samples_per_baboon2[samples_per_baboon2['baboon_id'] == baboon]['sample'])
    my_bc = bray_curtis_df.loc[samples[0],samples[0]]
    subsequented_dists = [my_bc.iloc[i, i+1] for i in range(len(my_bc) - 1)]
    print(np.mean(subsequented_dists))

print("finished bray curtis")
# the average distances are larger than expected

# Sample data: Distance matrix
distances = bray_curtis_matrix


# Calculate Bray-Curtis distance matrix
sorted_microbiome = pd.DataFrame(combined_df[microbiome.columns])
bray_curtis_distances = pdist(sorted_microbiome.iloc[:, 1:], metric='braycurtis')
bray_curtis_matrix = squareform(bray_curtis_distances)

# Display the distance matrix
bray_curtis_df = pd.DataFrame(bray_curtis_matrix, index=sorted_microbiome.index, columns=sorted_microbiome.index)
print(bray_curtis_df)

# choose baboons with most samples
sample_counts = sample_counts.sort_values(by='sample')
most_frequent = sample_counts.tail(2)
baboon_ids = most_frequent.index

choose = list(microbiome.columns)
choose = choose + ['baboon_id']
micro = pd.DataFrame(combined_df[choose])

micro = micro.loc[micro['baboon_id'] == 4].drop(['baboon_id', 'sample'], axis=1)
bray_curtis_distances = pdist(micro, metric='braycurtis')
bray_curtis_matrix = squareform(bray_curtis_distances)

# Display the distance matrix
bray_curtis_df = pd.DataFrame(bray_curtis_matrix)
print(bray_curtis_df)

print(combined_df.loc[combined_df['baboon_id'] == 4])

specific_baboon_id = 4
time_series_columns =microbiome.columns.drop('sample')
baboon_df = pd.DataFrame(combined_df.loc[combined_df['baboon_id'] == specific_baboon_id])

# for column in time_series_columns:
#     dta = baboon_df[column]
#     sm.graphics.tsa.plot_acf(dta.values.squeeze(), lags=30, auto_ylims=True, zero=False)
#     plt.title(f'Autocorrelation for baboon {specific_baboon_id} - {column}', fontsize=10)
#     plt.show()

plt.rcParams.update({'font.size': 14, 'xtick.labelsize': 14, 'ytick.labelsize': 14})

plt.figure(figsize=(10, 6))
sns.scatterplot(x='diet_PC1', y='diet_PC2', hue='group_num', data=combined_df, palette='viridis')
plt.title('PCA of FFQ Data Colored by Social Group', fontsize=16)
plt.xlabel('diet_PC1', fontsize=12)
plt.ylabel('diet_PC2', fontsize=12)
plt.legend(title='Social Group', fontsize=12, title_fontsize=14)
plt.savefig('pca_ffq_social_grp.pdf')

# Plot the first two PCs colored by seasonality
plt.figure(figsize=(10, 6))
sns.scatterplot(x='diet_PC1', y='diet_PC2', hue='season', data=combined_df, palette='viridis')
plt.title('PCA of FFQ Data Colored by Seasonality', fontsize=16)
plt.xlabel('diet_PC1', fontsize=12)
plt.ylabel('diet_PC2', fontsize=12)
plt.legend(title='Season', fontsize=12, title_fontsize=14)
plt.savefig('pca_ffq_seasonality.pdf')

plt.figure(figsize=(10, 6))
sns.scatterplot(x='diet_PC1', y='diet_PC2', hue='month', data=combined_df, palette='viridis')
plt.title('PCA of FFQ Data Colored by Month', fontsize=16)
plt.xlabel('diet_PC1', fontsize=12)
plt.ylabel('diet_PC2', fontsize=12)
plt.legend(title='Month', fontsize=12, title_fontsize=14)
plt.savefig('pca_ffq_month.pdf')

# We can see two clusters - the wet and the dry season . No appearent clustering by social groups. We wanted to check
# whether we can see better clustering in months division then by seasons devision, but we don't see firm clusters like
# in the seasons devision.

# print(bray_curtis_df)
# pcoa_coords = pcoa_results.samples
#
# # Calculate the explained variance ratio
# explained_variance_ratio = pcoa_results.eigvals / pcoa_results.eigvals.sum()
#
# # Print the explained variance ratio for the first two principal coordinates
# print(f'Explained variance by PC1: {explained_variance_ratio[0]*100:.2f}%')
# print(f'Explained variance by PC2: {explained_variance_ratio[1]*100:.2f}%')


# Initialize a list to store dissimilarities
dissimilarities = []

# Group by 'baboon_id'
cols = list(microbiome.columns) + ['baboon_id']
grouped = combined_df.groupby('baboon_id')

# Iterate over each group
for baboon_id, group in grouped:
    group = group.sort_values(by='collection_date')
    previous_sample = None

    for index, row in group.iterrows():
        if previous_sample is not None:
            actual = row[microbiome.columns].values
            predicted = previous_sample
            dissimilarity = braycurtis(actual, predicted)
            dissimilarities.append(dissimilarity)

        previous_sample = row[microbiome.columns].values


# Calculate the average Bray-Curtis dissimilarity
average_dissimilarity = sum(dissimilarities) / len(dissimilarities)
print(f'Average Bray-Curtis Dissimilarity: {average_dissimilarity}')

# לכל בבון מחושב המרחק בין הדגימה הנוכחית לקודמת ובסוף עושים ממוצע

df = combined_df.copy()
df['timestamp'] = df['collection_date'].values.astype(float)

microbiomee_columns = microbiome.columns.drop('sample')
ffq_columns = metadata.columns.drop(['baboon_id', 'sample', 'collection_date'])


# Initialize a list to store dissimilarities
dissimilarities = []

# Group by 'baboon_id'
grouped = df.groupby('baboon_id')

# Iterate over each group
for baboon_id, group in grouped:
    group = group.sort_values(by='collection_date')
    previous_sample = None

    # Naive prediction
    for index, row in group.iterrows():
        if previous_sample is not None:
            actual = row[microbiomee_columns].values
            predicted = previous_sample
            dissimilarity = braycurtis(actual, predicted)
            dissimilarities.append(dissimilarity)

        previous_sample = row[microbiomee_columns].values

# Calculate the average Bray-Curtis dissimilarity for naive prediction
average_dissimilarity_naive = sum(dissimilarities) / len(dissimilarities)
print(f'Average Bray-Curtis Dissimilarity (Naive): {average_dissimilarity_naive}')

def bray_curtis_dissimilarity(sample1, sample2):
    # Ensure both samples have the same shape
    if sample1.shape != sample2.shape:
        raise ValueError("Both samples must have the same shape.")

    # Initialize an array to store the dissimilarity for each column
    dissimilarity_scores = []

    # Iterate over each column (axis=0)
    for col1, col2 in zip(sample1.T, sample2.T):
        # Calculate sum of lesser counts (C_ij)
        C_ij = np.sum(np.minimum(col1, col2))

        # Total counts for both samples in this column
        S_i = np.sum(col1)
        S_j = np.sum(col2)

        # Calculate Bray-Curtis dissimilarity for this column
        if S_i + S_j == 0:
            dissimilarity_scores.append(0)  # Handle edge case where both are zero
        else:
            dissimilarity_scores.append(1 - (2 * C_ij) / (S_i + S_j))

    return np.array(dissimilarity_scores)

plt.figure(figsize=(15, 10))
sns.boxplot(data=microbiome)
plt.xticks(rotation=90)
plt.title('Distribution of Bacteria Percentages')
plt.savefig('bacteria_distribution.pdf')
from sklearn.decomposition import PCA
microbiome_data = pd.DataFrame(microbiome.drop('sample', axis=1))
# Apply PCA to microbiome data
pca = PCA(n_components=5)  # Choose top 5 components
pca.fit(microbiome_data)

# Get the bacteria contributing most to the first principal component
important_bacteria_pca = microbiome_data.columns[np.argsort(pca.components_[0])[:5]]  # Top 5 bacteria for PC1
print("Important bacteria based on PCA:", important_bacteria_pca)

# Calculate the mean abundance of each bacteria across all samples
#could be biased since the number of samples per baboon differs
mean_abundance = microbiome_data.mean(axis=0)

# Sort bacteria by their mean abundance (high to low)
important_bacteria_abundance = mean_abundance.sort_values(ascending=False).index[:5]  # Select top 5 most prevalent bacteria
print("Most prevalent bacteria:", important_bacteria_abundance)

# Calculate the variance of each bacteria across all samples
variance = microbiome_data.var(axis=0)

# Sort bacteria by variance (high to low)
important_bacteria_var = variance.sort_values(ascending=False).index[:5]  # Select top 5 bacteria with highest variance
print("Bacteria with highest variance:", important_bacteria_var)

important_bacteria = list(set(important_bacteria_pca) | set(important_bacteria_abundance) | set(important_bacteria_var))
print(important_bacteria)