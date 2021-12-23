# ML4science_project
Classification of ordinal outcomes for the analysis of injury severity using machine learning methods.

All this code was run in Python 3.8.11 with the following libraries:
Numpy 1.19.2
Pandas 1.3.4
Sklearn 1.0.1
Pytorch 1.7.1

## Get started

In case some of the datasets (see below) are missing, generate the preprocessed versions with 

```
python preprocess.py
```

Run 

```
python run.py
```

to train an initial model with the parameters from params.py

## File overview:

- dataset
  - 1.08 Crash Data (detail) DD.csv # overview over features
  - age_binned_preprocessed.csv # preprocessed with age as categorical binned columns
  - age_binned.csv # original dataset with age as categorical binned columns
  - area.png
  - crashdata.csv # original, uncleaned dataset
  - preprocessed_data.csv # preprocessed dataset: mean imputed and standardized (output of preprocess.py)
  - tempe_cleaneddata.csv # cleaned dataset 
- cross_validation.py # functions for k-fold cross-validation
- data_exploration.ipynb # initial data analysis and overview
- evaluation.py # functions for model evaluation metrics
- model.py # models as subclasses of pytorch.nn.Module
- params.py # parameters for models + training
- preprocess.py # preprocessing pipeline: mean imputation, standardization, age binning
- README.md
- run.py # run training of model
