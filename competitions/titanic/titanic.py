# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
import os
for dirname, _, filenames in os.walk('data'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# Get train data
train_data = pd.read_csv("/Users/pgutbrod/Development/private/projects/kaggle/competitions/titanic/data/train.csv")
#print(train_data.head())

# Get test data
test_data = pd.read_csv("/Users/pgutbrod/Development/private/projects/kaggle/competitions/titanic/data/test.csv")
#print(test_data.head())

# Analyse data
women = train_data.loc[train_data.Sex == 'female']['Survived']
men = train_data.loc[train_data.Sex == 'male']['Survived']

survival_rate_women = sum(women)/len(women)
survival_rate_men = sum(men)/len(men)

print(f'{survival_rate_women:.2%} of women survived')
print(f'{survival_rate_men:.2%} of men survived')

# Use RandomForest to predict
from sklearn.ensemble import RandomForestClassifier

y = train_data['Survived']

features = ['Pclass', 'Sex', 'SibSp', 'Parch']
X = pd.get_dummies(train_data[features])
X_test = pd.get_dummies(test_data[features])

rf_model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
rf_model.fit(X,y)
predictions = rf_model.predict(X_test)


output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
output.to_csv(
    '/Users/pgutbrod/Development/private/projects/kaggle/competitions/titanic/submissions/my_submission.csv', index=False)

