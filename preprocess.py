import pandas as pd

input_data = pd.read_csv('dataset/tempe_cleaneddata.csv', sep='\t', index_col=0)

cats, bins = pd.cut(input_data['age'], bins=[-1,1] + [x for x in range(25,95,10)], retbins=True)
dummies = pd.get_dummies(cats, prefix='age')

age_binned = pd.concat([input_data.drop(columns=['age']), cats], axis=1)

age_binned.to_csv('dataset/age_binned.csv', sep='\t')

# impute mean for age
mean_age = input_data['age'].mean()
input_data.loc[input_data['age'] == 0, 'age_isna'] = 1
input_data.loc[input_data['age'] == 0, 'age'] = mean_age
input_data['age_isna'] = input_data['age_isna'].fillna(0)

input_data['age'] = (input_data['age'] - input_data['age'].mean())/input_data['age'].std()
input_data['total_injuries'] = (input_data['total_injuries'] - input_data['total_injuries'].mean())/input_data['total_injuries'].std()

input_data.to_csv('dataset/preprocessed_data.csv', sep='\t')

age_binned['total_injuries'] = (age_binned['total_injuries'] - age_binned['total_injuries'].mean())/age_binned['total_injuries'].std()

age_binned.to_csv('dataset/age_binned_preprocessed.csv', sep='\t')