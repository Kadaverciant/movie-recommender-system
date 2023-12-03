# Movie Recommender System

by Vsevolod Klyushev IU F23 / D20-AI / v.klyushev@innopolis.university

Archive with checkpoints is available [here](https://drive.google.com/drive/folders/1-XvIDolm6SMfjJgAb-uc6leuoCncWer4?usp=sharing).

## Before start

Install all the packages from _requirements.txt_ using `pip install -r requirements.txt`

## Repository structure

```
movie-recommender-system
├── README.md               # The top-level README
│
├── data
│   ├── external            # Data from third party sources
│   ├── interim             # Intermediate data that has been transformed.
│   └── raw                 # The original, immutable data
│
├── models                  # Trained and serialized models, final checkpoints
│
├── notebooks               #  Jupyter notebooks.
│ 
├── references              # Data dictionaries, manuals, and all other explanatory materials.
│
├── reports
│   ├── figures             # Generated graphics and figures to be used in reporting
│   └── final_report.md     # Report containing data exploration, solution exploration, training process, and evaluation
│
├── requirements.txt  # The requirements file for reproducing the analysis environment
│                      generated with pip freeze › requirements. txt'
|
└── benchmark
    ├── data                # dataset used for evaluation 
    └── evaluate.py         # script that performs evaluation of the given model
```

## Basic usage

This section describes how to use script for benchmark calculation from `benchmark/` folder.

Help messages is available with `-h` flag. For example, `python ./benchmark/evaluate.py -h` explains all the available flags and their purpose.

If you want to use default parameters, you can run just `python ./benchmark/evaluate.py`.

## Contacts

In case of any questions you can contact me via telegram **@Kiaver**.
