import pandas as pd


microbiome = pd.read_csv("train_data.csv")
metadata = pd.read_csv("train_metadata.csv")
test_microbiome = pd.read_csv("short_timeseries_data.csv")
test_metadata = pd.read_csv("test_metadata.csv")
# cleaning datatypes

microbiome['sample'] = microbiome['sample'].str.replace('sample_', '')
test_microbiome['sample'] = test_microbiome['sample'].str.replace('sample_', '')

metadata['sample'] = metadata['sample'].str.replace('sample_', '')
test_metadata['sample'] = test_metadata['sample'].str.replace('sample_', '')

metadata['baboon_id'] = metadata['baboon_id'].str.replace('Baboon_', '')
metadata['baboon_id'] = metadata['baboon_id'].astype(int)

test_metadata['baboon_id'] = test_metadata['baboon_id'].str.replace('Baboon_', '')
test_metadata['baboon_id'] = test_metadata['baboon_id'].astype(int)

microbiome.columns = microbiome.columns.str.replace('g_', '')
test_microbiome.columns = test_microbiome.columns.str.replace('g_', '')

# change season into number classes
metadata['season'] = metadata['season'].replace({'wet': 0, 'dry': 1})
metadata['sex'] = metadata['sex'].replace({'M': 0, 'F': 1})

test_metadata['season'] = test_metadata['season'].replace({'wet': 0, 'dry': 1})
test_metadata['sex'] = test_metadata['sex'].replace({'M': 0, 'F': 1})

# Ensure 'collection date' is in datetime format
metadata['collection_date'] = pd.to_datetime(metadata['collection_date'])
test_metadata['collection_date'] = pd.to_datetime(test_metadata['collection_date'])

# Extract numeric part from the 'group' column and convert to float
metadata['group_num'] = metadata['social_group'].str.extract(r'_(\d+\.\d+)').astype(float)
metadata = metadata.drop(['social_group'], axis=1)

test_metadata['group_num'] = test_metadata['social_group'].str.extract(r'_(\d+\.\d+)').astype(float)
test_metadata = test_metadata.drop(['social_group'], axis=1)

# make combined dataset
combined_df = pd.merge(microbiome, metadata, on='sample')
combined_df['sample'] = combined_df.index

combined_df_test = pd.merge(test_microbiome, test_metadata, on='sample')
combined_df_test['sample'] = combined_df_test.index

# Sort the DataFrame by baboon_id and collection date
metadata = metadata.sort_values(by=['baboon_id', 'collection_date'])
test_metadata = test_metadata.sort_values(by=['baboon_id', 'collection_date'])

combined_df = combined_df.sort_values(by=['baboon_id', 'collection_date'])
combined_df_test = combined_df_test.sort_values(by=['baboon_id', 'collection_date'])


metadata['day_of_month'] = metadata['collection_date'].dt.day
metadata['week_of_year'] = metadata['collection_date'].dt.isocalendar().week
metadata['year'] = metadata['collection_date'].dt.year


test_metadata['day_of_month'] = test_metadata['collection_date'].dt.day
test_metadata['week_of_year'] = test_metadata['collection_date'].dt.isocalendar().week
test_metadata['year'] = test_metadata['collection_date'].dt.year


features = ['sex', 'age', 'group_num', 'group_size', 'rain_month_mm', 'season', 'hydro_year', 'month', 'diet_PC1',
            'diet_PC2', 'diet_PC3', 'diet_PC4', 'diet_PC5', 'diet_PC6',
            'diet_PC7', 'diet_PC8', 'diet_PC9', 'diet_PC10', 'diet_PC11',
            'diet_PC12', 'diet_PC13'] + microbiome.columns.tolist()
# we removed sample and baboon ID since they obviously don't affect any results
