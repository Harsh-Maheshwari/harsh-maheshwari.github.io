---
title: Good Python Libraries
date: 2022-05-04 19:25:35
description:
tags: 
 - python_libraries 
---

## Trading
- [quantstats](https://github.com/ranaroussi/quantstats)
- [alphalens-reloaded](https://github.com/stefan-jansen/alphalens-reloaded)
- [alphalens-reloaded](https://github.com/stefan-jansen/alphalens-reloaded)
- [tsfresh](https://tsfresh.readthedocs.io/en/latest/text/forecasting.html#parameters-and-implementation-notes)
- [scalecast](https://scalecast.readthedocs.io/en/latest/)

## Python libraries
- https://mlops.toys/data-versioning
- https://madewithml.com/courses/mlops/labeling/
- https://mlflow.org/docs/latest/index.html
- https://umap-learn.readthedocs.io/en/latest/
- https://docs.feast.dev/
- https://www.timvink.nl/reproducible-reports-with-mkdocs/

## GitHub pages
- [ml-tooling/best-of-python](https://github.com/ml-tooling/best-of-python)
- [ml-tooling/best-of-ml-python](https://github.com/ml-tooling/best-of-ml-python)
- [ml-tooling/ml-workspace](https://github.com/ml-tooling/ml-workspace)


## [Kedro DVC setup](https://youtu.be/ZTrFpeTCnc0)
```
kedro new
```
```
kedro mlflow init
kedro docker init
echo "dvc" >> src/requirements.txt
pip install -r src/requirements.txt
```
```
dvc init
dvc add data/01_raw
git rm data/01_raw/.gitkeep
git add -u
git commit -m "spec the datasets"
dvc add data/01_raw
git add data/01_raw.dvc
git add data/.gitignore
git commit -m "Added input data to DVC"
```

