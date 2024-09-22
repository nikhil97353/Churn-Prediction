#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


df=pd.read_csv("Churn_data.csv")
df


# In[9]:


df.head()


# In[6]:


df.info()


# In[7]:


df.isnull().sum()


# # EDA

# In[26]:


print("\nSummary statistics of numerical features:")
df.describe()


# In[28]:


dataset.columns


# In[35]:


print(df.describe())


# In[36]:


processed_data = df.dropna()


# In[47]:


data_frame = processed_data.drop(columns=['HasCrCard',"IsActiveMember",'Exited','RowNumber'],axis=True)


# In[48]:


# Select only numeric columns
numeric_cols = data_frame.select_dtypes(include=['float64', 'int64']).columns

# Generate the pairplot
sns.pairplot(data_frame[numeric_cols])
plt.show()


# In[55]:





# In[56]:


df.nunique()


# In[57]:


X = df.drop(columns=['Exited']).values
y = df['Exited'].values


# In[58]:


X


# In[59]:


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder


# In[62]:


ct = ColumnTransformer(transformers=[('onehot',OneHotEncoder(),[4])],remainder='passthrough')
X = ct.fit_transform(X)


# In[64]:


from sklearn.preprocessing import StandardScaler
sta=StandardScaler()
X = sta.fit_transform(X)


# # Test size = 0.2 

# In[85]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[86]:


from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from xgboost import XGBClassifier


# In[87]:


classifiers = {}
classifiers['SGDClassifier'] = SGDClassifier()
classifiers['Logistic Regression'] = LogisticRegression()
classifiers['K-Nearest Neighbors'] = KNeighborsClassifier()
classifiers['Support Vector Machine'] = SVC()
classifiers['Decision Tree'] = DecisionTreeClassifier()
classifiers['Random Forest'] = RandomForestClassifier()
classifiers['XGBoost'] = XGBClassifier()


# In[88]:


for name, model in classifiers.items():
    print(f"Training {name}...")
    model.fit(X_train, y_train)
    print(f"Evaluating {name}...")
    y_pred = model.predict(X_test)
    print(f"Classification Report for {name}:")
    print(classification_report(y_test, y_pred))
    print("*"*20)


# In[77]:


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC AUC']

for name, model in classifiers.items():
    print(f"Evaluating {name}...")
    y_pred = model.predict(X_test)

    # Calculate evaluation metrics
    scores = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1-Score': f1_score(y_test, y_pred),
        'ROC AUC': roc_auc_score(y_test, y_pred)
    }
    
    # Print evaluation metrics
    print(f"Evaluation Metrics for {name}:")
    for metric in metrics:
        print(f"{metric}: {scores[metric]}")
    
    # Confusion Matrix
    print(f"Confusion Matrix for {name}:")
    print(confusion_matrix(y_test, y_pred))
    
    print("="*50)


# In[91]:


from sklearn.neural_network import MLPClassifier

# Adding MLPClassifier (Neural Network) to classifiers
classifiers['Neural Network'] = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42)

# Train and evaluate models including Neural Network
for name, model in classifiers.items():
    print(f"Training {name}...")
    model.fit(X_train, y_train)
    print(f"Evaluating {name}...")
    
    if hasattr(model, "predict_proba"):  # If the classifier supports predict_proba
        y_probs = model.predict_proba(X_test)[:, 1]
    else:
        y_probs = model.decision_function(X_test)  # For models like SVM that use decision_function

    y_pred = model.predict(X_test)

    print(f"Classification Report for {name}:")
    print(classification_report(y_test, y_pred))
    print("*" * 20)

    # Plot ROC and Precision-Recall curves
    plot_roc_pr_curves(y_test, y_probs, name)


# In[ ]:




