# Titanic Survival Prediction Project

## Overview

This project involves analyzing the Titanic dataset to perform exploratory data analysis (EDA), data visualization, and building a logistic regression model to predict the survival of passengers.

## Steps Involved

1. **Data Exploration and Visualization:**
   - Used summary statistics and visualizations to understand data distribution and relationships.
   - Visualized survival rates by gender, class, and age distribution using `seaborn` and `matplotlib`.

2. **Data Preprocessing:**
   - Handled missing values by filling with median for numerical features and mode for categorical features.
   - Encoded categorical variables using one-hot encoding.
   - Selected relevant features for modeling.

3. **Model Building:**
   - Split the data into training and testing sets.
   - Built a logistic regression model using `scikit-learn`.
   - Evaluated the model's performance using accuracy score, confusion matrix, and classification report.

## Key Code Snippets

### Data Visualization

```python
import seaborn as sns
import matplotlib.pyplot as plt

# Survival Rate by Gender
sns.countplot(data=titanic_df, x='Survived', hue='Sex')
plt.title('Survival Rate by Gender')
plt.show()

# Survival Rate by Class
sns.countplot(data=titanic_df, x='Survived', hue='Pclass')
plt.title('Survival Rate by Class')
plt.show()

# Age Distribution of Passengers
sns.histplot(titanic_df['Age'], bins=30, kde=True)
plt.title('Age Distribution of Passengers')
plt.show()

 
