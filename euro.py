import pandas as pd

import numpy as np
from pathlib import Path

from sklearn.model_selection import train_test_split, cross_val_score
import sklearn
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from collections import Counter
from imblearn.over_sampling import ADASYN
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder

filename = Path(".src/results.csv")

df = pd.read_csv(filename, sep=";")

new_df=pd.concat([df['date'],df['home_team'],df['away_team'],df['city'],df['country'],df['result']],axis=1)

target_feature = 'home_team'
encoder = LabelEncoder()
encoded_values = encoder.fit_transform(new_df[target_feature].values)
new_df["home_team_code"] = pd.Series(encoded_values, index=new_df.index)

target_feature = 'away_team'
encoded_values = encoder.transform(new_df[target_feature].values)
new_df["away_team_code"] = pd.Series(encoded_values, index=new_df.index)

target_feature = 'city'
encoder = LabelEncoder()
encoded_values = encoder.fit_transform(new_df[target_feature].values)
new_df["city_code"] = pd.Series(encoded_values, index=new_df.index)

target_feature = 'country'
encoder = LabelEncoder()
encoded_values = encoder.fit_transform(new_df[target_feature].values)
new_df["country_code"] = pd.Series(encoded_values, index=new_df.index)

new_df['year'] = pd.DatetimeIndex(df['date']).year
new_df['month'] = pd.DatetimeIndex(df['date']).month
new_df['day'] = pd.DatetimeIndex(df['date']).day


new_df2=pd.concat([new_df['day'],new_df['month'],new_df['home_team_code'],new_df['away_team_code'],new_df['result']],axis=1)
new_df2= new_df2.sample(frac =1, random_state=42)
new_df3= new_df2.drop(["result"],axis=1)

data_train, data_test, y_train, y_true = \
    train_test_split(new_df3, new_df2['result'], test_size=0.2, random_state= 42)

counter = Counter(y_train)
print(counter)
oversample = ADASYN()
X, y = oversample.fit_resample(data_train, y_train)
counter = Counter(y)
print(counter)

d = preprocessing.normalize(X)
scaled_df = pd.DataFrame(d)
scaled_df.head()

t = preprocessing.normalize(data_test)
scaled_test = pd.DataFrame(t)
scaled_test.head()

classifier = RandomForestClassifier(n_estimators=1000)

model = classifier.fit(scaled_df, y)
y_test = model.predict(scaled_test)

confmat=confusion_matrix(y_true, y_test)
ticks=np.linspace(1, 2,num=2)
plt.imshow(confmat, interpolation='none')
plt.colorbar()
plt.xticks(ticks,fontsize=6)
plt.yticks(ticks,fontsize=6)
plt.grid(True)
plt.show()

sklearn.metrics.accuracy_score(y_true, y_test)


