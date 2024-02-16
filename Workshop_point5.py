# ==============================================================================
#                               LIBRARY IMPORT
# ==============================================================================
# Data structure
import pandas as pd

# Graphics
import matplotlib.pyplot as plt
import seaborn as sns

# Datasets
from sklearn.datasets import load_digits

# # Data preparation
from sklearn.model_selection import train_test_split

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB

## Model evaluation
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

# ==============================================================================
#                          WORKSHOP SOLUTION - POINT 5
# ==============================================================================

# 5) Train a naive logistic regression on raw MNIST images to distinguish 
# between 0s and 8s. We are calling this our baseline. What can you tell about 
# the baseline performance?
# ==============================================================================
# Load dataset
mnist = load_digits()

# Define arrays X: data matrix, y: classification target
X = pd.DataFrame(mnist.data)
y = pd.DataFrame(mnist.target)

# Filter data by selecting classes 0s and 8s
y_filtered = y[y[0].isin([0,8])]
X_filtered = X.iloc[y_filtered.index]

# Training and test datasets
X_train, X_test, y_train, y_test = train_test_split(X_filtered,
                                                    y_filtered, 
                                                    test_size=0.3, 
                                                    random_state = 123,
                                                    shuffle = True)

# Create linear regression object
# model = LogisticRegression()
model = GaussianNB()

# Train the model using the training sets
model.fit(X_train, y_train.squeeze())

# Make predictions using the testing set
y_pred = model.predict(X_test)

# Model evaluateion
accuracy_score(y_test, y_pred)

# Confusion matrix
conf_mat = confusion_matrix(y_test, y_pred)

print(metrics.classification_report(y_test, y_pred))

fig, ax = plt.subplots(figsize=(6,4))
sns.heatmap(conf_mat
            , annot=True
            , fmt='d'
            , cmap = 'mako'
           )
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title("Confusion matrix")
plt.show()

print("The baseline performance was very good. Precision, \
recall anf f1-score are above 96% while in the Confusion \
matrix the failures are minimal")

