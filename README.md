# Poland-Bankruptcy
According to company data provided by [UCI university](https://archive.ics.uci.edu/ml/datasets/Polish+companies+bankruptcy+data), Use the classification model to predict whether a company is able to be bankrupted
## Overview:
The dataset is about bankruptcy prediction of Polish companies. The data was collected from Emerging Markets Information Service, which is a database containing information on emerging markets around the world. The bankrupt companies were analyzed in *the period 2000-2012*, while the still operating companies were evaluated from *2007 to 2013*.

About the attribute information, please refer to [data dictionary.ipynb](https://github.com/Trihamhochoi/Poland-Bankruptcy/blob/main/data_dictionary.ipynb)
## Introduction:
My model is divided into two parts: [EDA](https://github.com/Trihamhochoi/Poland-Bankruptcy/blob/main/EDA.ipynb) and [the classification](https://github.com/Trihamhochoi/Poland-Bankruptcy/blob/main/Classification_Decision_tree.ipynb) by 2 models (Random Forest and Gradient Booster).

Simultaneously, I will compare the performance and running time of  2 models and select the appropriate model.In addition, I will validate the model using the confusion report to tune tune hyperparameters effectively.

## How to use my code:
You can use my code to create your personal model as well as improve it more professionally. Such as:
- If you have data of other countries, you could try to personalize it and predict the results.
- Or you could build a dasboard to visualize some metrics.

## Machine Learning algorithms:
- [Decision Tree](https://en.wikipedia.org/wiki/Decision_tree)
- [Gradient booster](https://en.wikipedia.org/wiki/Gradient_boosting)

**My Conclusion:** 
In my case, after using cross validation for both algorithsms, I see Random Forest not only saves slightly more time but also gives better score than the other.   
For this reason, we recommend to use Randomforest for this model.
 
## PART1: Exploratory Data Analysis:
After cleaning and preproccessing data, I chose some attribution to analysis. 
1. `feat_27:` **profit on operating activities / financial expenses**

