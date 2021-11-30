import pandas as pd

input_data = pd.read_csv('dataset/tempe_cleaneddata.csv', sep='\t', index_col=0)

# impute mean for age
mean_age = input_data['age'].mean()
input_data.loc[input_data['age'] == 0, 'age_isna'] = 1
input_data.loc[input_data['age'] == 0, 'age'] = mean_age
input_data['age_isna'] = input_data['age_isna'].fillna(0)

input_data['age'] = (input_data['age'] - input_data['age'].mean())/input_data['age'].std()
input_data['total_injuries'] = (input_data['total_injuries'] - input_data['total_injuries'].mean())/input_data['total_injuries'].std()

input_data.to_csv('dataset/preprocessed_data.csv', sep='\t')