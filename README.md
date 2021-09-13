# Breast-Cancer-Prediction

#### Goal: To develop a POC using Flask, HTML and CSS for predicting whether a person is suffering from Breast Cancer or not, implementing Machine Learning algorithm.

### About the Data set: 
This is a machine learning project where we will predict whether a person is suffering Breast Cancer or not. The dataset was downloaded from [Kaggle](https://www.kaggle.com/uciml/breast-cancer-wisconsin-data).

Attribute Information:

1) ID number
2) Diagnosis (M = malignant, B = benign)
3-32)

Ten real-valued features are computed for each cell nucleus:

a) radius (mean of distances from center to points on the perimeter)
b) texture (standard deviation of gray-scale values)
c) perimeter
d) area
e) smoothness (local variation in radius lengths)
f) compactness (perimeter^2 / area - 1.0)
g) concavity (severity of concave portions of the contour)
h) concave points (number of concave portions of the contour)
i) symmetry
j) fractal dimension ("coastline approximation" - 1)

The mean, standard error and "worst" or largest (mean of the three
largest values) of these features were computed for each image,
resulting in 30 features. For instance, field 3 is Mean Radius, field
13 is Radius SE, field 23 is Worst Radius.

All feature values are recoded with four significant digits.

Missing attribute values: none

Class distribution: 357 benign, 212 malignant

The target column says whether the person is having the disease or not based on the predictor variables. A binary classification problem statement.

### Project Description: 
After loading the dataset("breast_cancer.csv") the first step was to perform an extensive Exploratory Data Analysis(EDA).
Initially, the two features "Unnamed: 32" and "id" was dropped as "Unnamed: 32" has 569 of nan values and "id" was not that useful.
Then countplots were created for the target feature to check whether the dataset is balanced or not.
It was a balanced dataset. Violin plots were created for outliers detection. Then Joint plots for features concavity_worst and concave points_worst were created and it was found that these two features were highly correlated.
Kernel Density Estimator plots and scatter plots were created for the features 'radius_worst','perimeter_worst','area_worst' for better understanding about the relationship between these three features. Then, the dataset was divided into independent(x) features and Dependent(Y) features for the purpose of Data Analysis. A correlation heatmap was made to check the correlation between all the independent features.

The second step was to perform Feature Engneering. The dataset was divided into independent(X) and dependent(y) features. Label Encoding was performed on the target feature. 1 for Malignant(M) and 0 for Benign(B).
Feature Scaling was performed using Sklearn's StandardScaler. Scaling was performed on 'concave points_mean','area_mean','radius_mean','perimeter_mean','concavity_mean' .

The third step was Feature Selection. Features were selected manually based on domain knowledge.
concave points_mean','area_mean','radius_mean','perimeter_mean','concavity_mean' were the features that got selected.

The Forth step was Model Building. Train test split was performed for getting the train and test datasets.
Logistic Regression was applied on the training data after testing with other Machine Learning algorithmns.
Predicton and validaion was performed on the test dataset.

The fifth step was to perform Hyperparameter Optimization on our model. A range of parameters for "penalty", "C", "solver", "max_iter" was selected and passed through GridSearchCV.
The model was then fitted with the best parameters. The main aim was to reduce the False Positives and the False Negatives. Model performed really good and validated based on classification report, confusion matrix and accuracy score.

The final step was to save the model as a pickle file to reuse it again for the Deployment purpose. Joblib was used to dump the model at the desired location.

The "Breast Cancer Prediction.ipynb" file contains all these informations.

Deployment Architecture: The model was deployed locally (port 5000). The backend part of the application was made using Flask and for the frotend part HTML and CSS was used.
I have not focussed much on the frontend as I am not that good at it. The file "app.py" contains the entire flask code and inside the templates folder, "cancer.html" contains the homepage and "result.html" contains the result page. 

### Installation:
The Code is written in Python 3.7.3 If you don't have Python installed you can find it [here](https://www.python.org/downloads/). If you are using a lower version of Python you can upgrade using the pip package, ensuring you have the latest version of pip. To install the required packages and libraries, run this command in the project directory after [cloning](https://www.howtogeek.com/451360/how-to-clone-a-github-repository/) the repository:

##### 1. First create a virtual environment by using this command:
```bash
conda create -n myenv python=3.7
```
##### 2. Activate the environment using the below command:
```bash
conda activate myenv
```
##### 3. Then install all the packages by using the following command
```bash
pip install -r requirements.txt
```
##### 4. Then, in cmd or Anaconda prompt write the following code:
```bash
python app.py
```
##### Make sure to change the directory to the root folder.  

### A Glimpse of the application
![Screenshot (156)](https://user-images.githubusercontent.com/75041273/133083338-8ff0f2fa-5ed9-4840-af4a-510870b60d2a.png)
![Screenshot (157)](https://user-images.githubusercontent.com/75041273/133083362-e9761145-3f78-4250-8344-836408969a26.png)
![Screenshot (155)](https://user-images.githubusercontent.com/75041273/133083388-492c3c4a-3b9f-4c8e-9270-d7a02f7bd352.png)

### Further Changes to be Done
- [ ] Including more features.
- [ ] Deploying the Web Application on Cloud.
     - [ ] Google Cloud 
     - [ ] Azure
     - [ ] Heroku
     - [ ] AWS

### Technology Stack

<img src="https://img.shields.io/badge/Python-FFD43B?style=for-the-badge&logo=python&logoColor=darkgreen" /> <img src="https://img.shields.io/badge/Numpy-777BB4?style=for-the-badge&logo=numpy&logoColor=white" /> <img src="https://img.shields.io/badge/Pandas-2C2D72?style=for-the-badge&logo=pandas&logoColor=white" /> ![Seaborn](https://img.shields.io/badge/Seaborn-%230C55A5.svg?style=for-the-badge&logo=seaborn&logoColor=%white)  <img src="https://img.shields.io/badge/scikit_learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" /> <img src="https://img.shields.io/badge/Jupyter-F37626.svg?&style=for-the-badge&logo=Jupyter&logoColor=white" /> <img src="https://img.shields.io/badge/conda-342B029.svg?&style=for-the-badge&logo=anaconda&logoColor=white"/> <img src="https://img.shields.io/badge/Kaggle-20BEFF?style=for-the-badge&logo=Kaggle&logoColor=white" />  <img src="https://img.shields.io/badge/matplotlib-342B029.svg?&style=for-the-badge&logo=matplotlib&logoColor=white"/> <img src="https://img.shields.io/badge/Flask-000000?style=for-the-badge&logo=flask&logoColor=white" /> <img src="https://img.shields.io/badge/Spyder-838485?style=for-the-badge&logo=spyder%20ide&logoColor=maroon" />
