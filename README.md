##Optimally combining classifiers for semi-supervised learning

We propose a new semi-supervised method combing Xgboost and transductive support vector machine.

Experiments on 14 UCI data 




### Requirements
- Python 3.6+
- pandas
- matplotlib
- numpy
- Pycharm



### Train

Obtain the visualization of the real data by T-SNE

```
python DataDistribution.py
```

Compare the diversity of Xgboost, TSVM, DecisionTree

```
python compare_diversity.py
```
Run the proposed method for 14 real data

```
python new_algorithm_test
```
### Results (Accuracy)
| Data | cjs | hill | segment | wdbc| steel |analcat |synthetic |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|Accuracy |0.98  | 0.499 | 0.925| 0.954 |0.649|0.993|0.92|
### Results (Accuracy)
| Data | vehicle | german | gina | madelon| texture |gas-grift |dna |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|Accuracy |0.625  | 0.716 | 0.857| 0.543 |0.953|0.965|0.911|

