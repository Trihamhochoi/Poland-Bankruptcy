# Poland-Bankruptcy
According to company data provided by [UCI university](https://archive.ics.uci.edu/ml/datasets/Polish+companies+bankruptcy+data), Use the classification model to predict whether a company is able to be bankrupted
# Overview:
The dataset is about bankruptcy prediction of Polish companies. The data was collected from Emerging Markets Information Service, which is a database containing information on emerging markets around the world. The bankrupt companies were analyzed in *the period 2000-2012*, while the still operating companies were evaluated from *2007 to 2013*.
Basing on the collected data five classification cases were distinguished, that depends on the forecasting period:
- 1stYear the data contains financial rates from 1st year of the forecasting period and corresponding class label that indicates bankruptcy status after 5 years. 
  - There are 7027 instances, which are comprised of 271 bankrupt companies and 6756 non-bankrupt companies.

- 2ndYear the data contains financial rates from 2nd year of the forecasting period and corresponding class label that indicates bankruptcy status after 4 years. 
  - There are 10173 instances, which are comprised of 400 bankrupt companies and 9773 non-bankrupt companies.
  
- 3rdYear the data contains financial rates from 3rd year of the forecasting period and corresponding class label that indicates bankruptcy status after 3 years. 
  - There are 10503 instances, which are comprised of 495 bankrupt companies and 10008 non-bankrupt companies.
  
- 4thYear the data contains financial rates from 4th year of the forecasting period and corresponding class label that indicates bankruptcy status after 2 years. 
  - There are 9792 instances, which are comprised of 515 bankrupt companies and 9277 non-bankrupt companies.

About the attribute information, please refer to [data dictionary.ipynb](https://github.com/Trihamhochoi/Poland-Bankruptcy/blob/main/data_dictionary.ipynb)

# Introduction:
My model is divided into two parts: [EDA](https://github.com/Trihamhochoi/Poland-Bankruptcy/blob/main/EDA.ipynb) and [the classification](https://github.com/Trihamhochoi/Poland-Bankruptcy/blob/main/Classification_Decision_tree.ipynb) by 2 models (Random Forest and Gradient Booster).

Simultaneously, I will compare the performance and running time of  2 models and select the appropriate model. As well as I will validate the model by the confusion report.  
