# Industrial Copper Modeling Application

## Overview
This application is designed to predict the selling price and status of copper products based on various input parameters. The project is implemented using Python, and it utilizes machine learning models built with Scikit-Learn.

## Requirements
- Python 3.x
- Jupyter Notebook (for training models)
- Visual Studio Code (for running the Streamlit app)
- Required Libraries:
  - pandas
  - numpy
  - scikit-learn
  - streamlit
  - pickle
  - re
  - warnings

## Model Training (Jupyter Notebook)
### Regressor (Predict Selling Price)
```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
```

### Classifier (Predict Status)
```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelBinarizer
```

## Streamlit Application (VS Code)
### Running the Application
```sh
streamlit run app.py
```

### Features
- **Predict Selling Price**: 
  - Takes input parameters like Status, Item Type, Country, Application, Product Reference, Quantity Tons, Thickness, Width, and Customer ID.
  - Uses a trained Decision Tree Regressor model to predict the selling price.
- **Predict Status**:
  - Uses parameters like Quantity Tons, Thickness, Width, Customer ID, Selling Price, and Item Type.
  - Uses a trained Decision Tree Classifier model to predict whether the status is "Won" or "Lost".

### Application UI
- The UI is built using Streamlit.
- Two tabs:
  - **PREDICT SELLING PRICE**: Allows users to enter required inputs and get a predicted selling price.
  - **PREDICT STATUS**: Takes relevant input and predicts whether a deal is "Won" or "Lost".

### Data Preprocessing
- StandardScaler is used for feature scaling.
- OneHotEncoder is applied for categorical variables.
- Log transformation is applied to relevant numeric features to normalize data.

### Model Deployment
- The trained models are saved using `pickle`.
- Models are loaded in Streamlit for inference.

## Author
App Created by **Ajay Kumar**

