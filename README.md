# ML4science_project
Classification of ordinal outcomes for the analysis of injury severity using machine learning methods.

## Get started

Run 

```
python run.py
```

to train an initial model

## File overview:

- dataset
  - 1.08 Crash Data (detail) DD.csv # overview over features
  - area.png
  - crashdata.csv # original, uncleaned dataset
  - tempe_cleaneddata.csv # cleaned dataset 
- data_exploration.ipynb # initial data analysis and overview
- model.py # models als subclasses of pytorch.nn.Module
- params.py # parameters for models + training
- README.md
- run.py # run training of model
