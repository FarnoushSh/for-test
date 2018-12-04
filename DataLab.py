import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)
from pandas.io.formats.style import Styler
from sklearn import preprocessing
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
import statsmodels.api as sm
from sklearn import metrics
plt.rc("font", size=14)
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectFromModel
from sklearn import model_selection
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier



train_data=pd.read_csv('TrainDataApo.csv',sep=';', engine='python')
test_data=pd.read_csv('TestDataApo.csv', sep=';', engine='python')
train_data.columns = train_data.columns.str.replace(' ', '_')
test_data.columns = test_data.columns.str.replace(' ', '_')
# train_data.rename(columns={ 'Schulabschlu�' : 'Schulabschluss'}, inplace=True)
# test_data.rename(columns={ 'Schulabschlu�' : 'Schulabschluss'}, inplace=True)
print(train_data.columns)

obj_data = train_data.select_dtypes(include=['object']).copy()
obj_col=obj_data.columns
for col in obj_col:
    print(obj_data.groupby([col]).size())

# def change_string(x):
#     return x.replace('Gr�nder',"Greunder").replace('Selbst�ndig','Selbsteandig')
#
# train_data['Art_der_Anstellung'] = train_data['Art_der_Anstellung'].map(lambda x: change_string(x))
# test_data['Art_der_Anstellung'] = test_data['Art_der_Anstellung'].map(lambda x: change_string(x))
# test_data.groupby('Art_der_Anstellung').size()



print(train_data.groupby('Zielvariable').mean())

#
# # Data Visualization
pd.crosstab(train_data.Monat,train_data.Zielvariable).plot(kind='bar')
plt.title('Zielvariable & Monat')
plt.xlabel('Monat')
plt.ylabel('Frequecy of Completion')
# plt.savefig('frequency_based_Monat')
plt.show()

pd.crosstab(train_data.Familienstand,train_data.Zielvariable).plot(kind='bar')
plt.title('Zielvariable & Familienstand')
plt.xlabel('Familienstand')
plt.ylabel('Frequecy of Completion')
# plt.savefig('frequency_based_Familienstand')
plt.show()


pd.crosstab(train_data.Ergebnis_letzte_Kampagne,train_data.Zielvariable).plot(kind='bar')
plt.title('Zielvariable & Ergebnis_letzte_Kampagne')
plt.xlabel('Ergebnis_letzte_Kampagne')
plt.ylabel('Frequecy of Completion')
# plt.savefig('frequency_based_Ergebnis_letzte_Kampagne')
plt.show()

pd.crosstab(train_data.Schulabschluß,train_data.Zielvariable).plot(kind='bar')
plt.title('Zielvariable & Schulabschluß')
plt.xlabel('Schulabschluß')
plt.ylabel('Frequecy of Completion')
# plt.savefig('frequency_based_Schulabschluß')
plt.show()


pd.crosstab(train_data.Art_der_Anstellung,train_data.Zielvariable).plot(kind='bar')
plt.title('Zielvariable and Jobs')
plt.xlabel('Job')
plt.ylabel('Frequecy of Completion')
plt.show()


pd.crosstab(train_data.Kredit,train_data.Zielvariable).plot(kind='bar')
plt.title('Zielvariable and Kredit status')
plt.xlabel('Kredit')
plt.ylabel('Frequecy of Completion')
plt.show()

pd.crosstab(train_data.Geschlecht,train_data.Zielvariable).plot(kind='bar')
plt.title('Zielvariable based on Geschlecht')
plt.xlabel('Geschlecht')
plt.ylabel('Frequecy of Completion')
# plt.savefig('frequency_based_Geschlecht')
plt.show()


pd.crosstab(train_data.Haus,train_data.Zielvariable).plot(kind='bar')
plt.title('Zielvariable and Haus status')
plt.xlabel('Haus')
plt.ylabel('Frequecy of Completion')
plt.show()


pd.crosstab(train_data.Kontaktart,train_data.Zielvariable).plot(kind='bar')
plt.title('Zielvariable and Kontakart status')
plt.xlabel('Kontaktart')
plt.ylabel('Frequecy of Completion')
plt.show()



plt.figure(figsize=(5,5))
sns.boxplot(x='Zielvariable',y='Anzahl_Kontakte_letzte_Kampagne',data=train_data)
plt.show()

# OUTLIERS

sns.FacetGrid(train_data, hue="Zielvariable", height=5) \
   .map(plt.scatter, "Dauer", "Alter") \
   .add_legend()
plt.show()

sns.FacetGrid(train_data, hue="Zielvariable", height=5) \
   .map(plt.scatter, "Kontostand", "Alter") \
   .add_legend()
plt.show()

sns.FacetGrid(train_data, hue="Zielvariable", height=5) \
   .map(plt.scatter, "Anzahl_Kontakte_letzte_Kampagne", "Alter") \
   .add_legend()
plt.show()


trolls_tr = train_data[(train_data.Kontostand >= 50000) | (train_data.Dauer >=2500)|(train_data.Anzahl_Kontakte_letzte_Kampagne >= 50)].index
train_data= train_data.drop(list(trolls_tr), axis=0)
trolls_ts = test_data[(test_data.Kontostand >= 50000) | (test_data.Dauer >=2500)|(train_data.Anzahl_Kontakte_letzte_Kampagne >= 50)].index
test_data= test_data.drop(list(trolls_ts), axis=0)


# # MissingValues
#
sns.heatmap(train_data.isnull(),yticklabels=False)
plt.show()


#After some examination, having or not having the Tag... variable does not change the result and so I drop it.
train_data.drop('Tage_seit_letzter_Kampagne',axis=1,inplace=True)


# encoding target values manually.
cleanup_nums = {'Zielvariable':{'nein':0,'ja':1}}
train_data.replace(cleanup_nums, inplace=True)

# change the continues variable kontostand into three categories (negative, positive and null)
train_data['Kontostand'] = ['Negative' if x < 0 else 'Null' if x == 0 else 'positive' if x > 0 else 'NaN' for x in train_data['Kontostand']]
test_data['Kontostand'] = ['Negative' if x < 0 else 'Null' if x == 0 else 'positive' if x > 0 else 'NaN' for x in test_data['Kontostand']]
#


#  Encoding categorical data using get_dummies
cat_vars_list=['Kontostand','Familienstand','Haus','Kredit','Ausfall_Kredit','Kontaktart',
          'Schulabschluß', 'Ergebnis_letzte_Kampagne','Art_der_Anstellung','Monat']
for var in cat_vars_list:
    cat_var_train='var'+'_'+ var
    cat_var_train = pd.get_dummies(train_data[var], prefix=var)
    cat_var_test= 'var' + '_' + var
    cat_var_test = pd.get_dummies(test_data[var], prefix=var)
    data1 = train_data.join(cat_var_train)
    data2 = test_data.join(cat_var_test)
    train_data = data1
    test_data = data2

data_vars_train = train_data.columns.values.tolist()
data_vars_test = test_data.columns.values.tolist()
keep_train = [i for i in data_vars_train if i not in cat_vars_list]
Train_dt = train_data[keep_train]
keep_test=[i for i in data_vars_test if i not in cat_vars_list]
Test_dt=test_data[keep_test]
print(Train_dt.columns)
print(Test_dt.columns)

# droping some unuseful variables
Train_dt=Train_dt.drop(['Tag','Stammnummer','Anruf-ID','Geschlecht'],axis=1)
Test_dt=Test_dt.drop(['Tag','Stammnummer','Anruf-ID','Geschlecht','Zielvariable'],axis=1)

Features=Train_dt.drop(['Zielvariable'], axis=1)
Features_name=Features.columns
Test_dt_col=Test_dt.columns

#Data Normalization
scaler=preprocessing.StandardScaler()
Features = scaler.fit_transform(Features)
Test_dt = scaler.fit_transform(Test_dt)

#Change arrays to pandas data frame
Features=pd.DataFrame(Features,columns=Features_name)
Test_dt=pd.DataFrame(Test_dt,columns=Test_dt_col)
Train_dt = Features.join(Train_dt['Zielvariable'])

# droping the NaN values
Train_dt=Train_dt.dropna()


# Data balancing using SMOTE
X = Train_dt[Features_name]
y = Train_dt[['Zielvariable']]
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
smt = SMOTE(random_state=0)
columns = x_train.columns
smt_data_X,smt_data_y = smt.fit_sample(x_train, y_train.values.ravel())
smt_data_X = pd.DataFrame(data=smt_data_X,columns=columns )
smt_data_y= pd.DataFrame(data=smt_data_y,columns=['Zielvariable'])
print("length of oversampled data is ",len(smt_data_X))
print("Number of no subscription in oversampled data",len(smt_data_y[smt_data_y['Zielvariable']==0]))
print("Number of subscription",len(smt_data_y[smt_data_y['Zielvariable']==1]))
print("Proportion of no subscription data in oversampled data is ",len(smt_data_y[smt_data_y['Zielvariable']==0])/len(smt_data_X))
print("Proportion of subscription data in oversampled data is ",len(smt_data_y[smt_data_y['Zielvariable']==1])/len(smt_data_X))
#

#

# #
# # # Feature Optimization
X = smt_data_X
y = smt_data_y['Zielvariable']

#

#
# Data Training (Logistic Regression. Random Forest Classsifier, Decision Tree Classifier)
selected_x = smt_data_X[Features_name]
X_train, X_test, y_train, y_test = train_test_split(selected_x, y, test_size=0.3, random_state=0)
logreg = LogisticRegression()
predicted = cross_val_predict(logreg, selected_x, y, cv=10)
logreg.fit(X_train, y_train)
#
y_pred_lr = logreg.predict(X_test)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test,y_test)))
print('Logistic-regression (cross_validation):',metrics.accuracy_score(y, predicted))

confusion_matrix = confusion_matrix(y_pred_lr,y_test)
print(confusion_matrix)

print(classification_report(y_pred_lr,y_test))
#
dtc = DecisionTreeClassifier()
dtc.fit(X_train, y_train)

y_pred = dtc.predict(X_test)
print('Accuracy of Decision Tree classifier on test set: {:.2f}'.format(dtc.score(X_test, y_test)))

rf = RandomForestClassifier()
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)
print('Accuracy of Random Forest classifier on test set: {:.2f}'.format(rf.score(X_test, y_test)))


# Comparing the results
MLA = []
Prediction = []
results = []
names = []
Z = [DecisionTreeClassifier(), LogisticRegression(), RandomForestClassifier()]
name = ["DecisionTreeClassifier", "LogisticRegression", "RandomForestClassifier"]

seed = 10
for i in range(0, len(Z)):
    model = Z[i]
    kfold = model_selection.KFold(n_splits=10, random_state=seed)
    cv_results = model_selection.cross_val_score(model, selected_x, y, cv=kfold, scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name[i], cv_results.mean(), cv_results.std())
    print(msg)

fig = plt.figure(figsize=(10, 10))
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(name)
plt.show()

#  AUC plot
plt.figure()

models = [
    {
        'label': 'Logistic Regression',
        'model': LogisticRegression(),
    },
    {
        'label': 'Logistic Regression, C=0.01',
        'model':  LogisticRegression(penalty='l2', C=0.01, random_state=0)
    },
    {
        'label': 'Logistic Regression, C=0.1',
        'model':  LogisticRegression(penalty='l2', C=0.1, random_state=0)
    },
    {
        'label': 'Random Forest',
        'model': RandomForestClassifier(),
    },
     {
        'label': 'Random Forest, max_depth : 30 ',
        'model': RandomForestClassifier(n_estimators=100, max_depth=30, random_state=0),
    },
    {
        'label': 'Decision Tree',
        'model': DecisionTreeClassifier(),
    },
    {
        'label': 'Decision Tree,max_depth=30',
        'model': DecisionTreeClassifier(max_depth=30,
                 max_features=10, min_samples_leaf=1,
                 min_samples_split=2,
                 random_state=0),
    }
]

for m in models:
    model = m['model']
    model.fit(X_train, y_train)
    y_pred_lr= model.predict(X_test)

    fpr, tpr, thresholds = metrics.roc_curve(y_test, model.predict_proba(X_test)[:, 1])

    auc = metrics.roc_auc_score(y_test, model.predict(X_test))

    plt.plot(fpr, tpr, label='%s ROC (area = %0.2f)' % (m['label'], auc))

plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Specificity(False Positive Rate)')
plt.ylabel('Sensitivity(True Positive Rate)')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()


# Extracting the result
winner_model=RandomForestClassifier(n_estimators=100, max_depth=30, random_state=0)
winner_model.fit(X_train, y_train)
Test_Data = Test_dt[Features_name]

prediction=pd.DataFrame(winner_model.predict(Test_Data),columns=['Zielvariable'])
proba=pd.DataFrame(list(winner_model.predict_proba(Test_Data)),columns=['zero','one'])
print(proba.head())

test_Id=test_data['Stammnummer']
predicted=proba['one']

my_submission = pd.DataFrame({'ID': test_Id, 'Expected': predicted})

my_submission.to_csv('submission.csv', index=False)
result=pd.read_csv('submission.csv')
print(result.head(30))