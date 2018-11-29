import pandas as pd
import warnings
warnings.filterwarnings("ignore")
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import seaborn as sns
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)
from sklearn import preprocessing
import sklearn
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
plt.rc("font", size=14)
from pandas.io.formats.style import Styler
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score



# ///////////////////Data Exploration////////////////////

train_data=pd.read_csv('TrainDataApo.csv', sep=';', engine='python')
test_data=pd.read_csv('TestDataApo.csv', sep=';', engine='python')
print(train_data.head())
print(train_data.info())
print(test_data.info())

#replacing white spaces with underline
train_data.columns = train_data.columns.str.replace(' ', '_')
test_data.columns = test_data.columns.str.replace(' ', '_')


# Object variables # categorical Data
obj_data = train_data.select_dtypes(include=['object']).copy()
obj_col=obj_data.columns
for col in obj_col:
    print(obj_data.groupby([col]).size())



##### Data Preprocessing and Visualization

# sns.countplot(x='Zielvariable',data=train_data,palette='hls')
# # plt.show()
# plt.savefig('Target_count_plot')
#
# # Our classes are imbalanced!
#
#
# # TO get some information about the relation between each variables
# print(train_data.groupby('Zielvariable').mean())
# print(train_data.groupby('Ausfall_Kredit').mean())
#
#
#
#
# fig = plt.figure()
#
# fig, axes = plt.subplots(nrows=2, ncols=2)
#
# pd.crosstab(train_data.Monat,train_data.Zielvariable).plot(kind='bar',ax=axes[0,0])
# pd.crosstab(train_data.Ergebnis_letzte_Kampagne,train_data.Zielvariable).plot(kind='bar',ax=axes[0,1])
# pd.crosstab(train_data.Kredit,train_data.Zielvariable).plot(kind='bar',ax=axes[1,0])
# pd.crosstab(train_data.Haus,train_data.Zielvariable).plot(kind='bar',ax=axes[1,1])
#
#
# plt.show()
#
# fig= plt.figure()
# fig, axes = plt.subplots(nrows=2, ncols=2)
# pd.crosstab(train_data.Art_der_Anstellung,train_data.Zielvariable).plot(kind='bar',ax=axes[1,1])
# pd.crosstab(train_data.Ausfall_Kredit,train_data.Zielvariable).plot(kind='bar',ax=axes[1,0])
# pd.crosstab(train_data.Kontaktart,train_data.Zielvariable).plot(kind='bar',ax=axes[0,0])
# pd.crosstab(train_data.Familienstand,train_data.Zielvariable).plot(kind='bar',ax=axes[0,1])
# plt.show()
#
# fig=plt.figure()
# fig, axes=plt.subplots(nrows=2, ncols=2)
# pd.crosstab(train_data.Geschlecht,train_data.Zielvariable).plot(kind='bar',ax=axes[0,0])
# pd.crosstab(train_data.Schulabschluß,train_data.Zielvariable).plot(kind='bar',ax=axes[1,0])
# plt.show()
#
# plt.figure(figsize=(10,10))
# sns.FacetGrid(train_data, hue="Zielvariable", size=5) \
#    .map(plt.scatter, "Dauer", "Alter") \
#    .add_legend()
# plt.show()
#
# plt.figure(figsize=(5,5))
# sns.boxplot(x='Zielvariable',y='Kontostand',data=train_data)
# plt.show()
#
#
#
# sns.distplot(train_data.Dauer)
# plt.show()
#
# sns.distplot(train_data.Kontotand)
# plt.show()

#
#
# Missing values
# ------------------------------------------------------
# # To display the places of missing values. It shows that the main missing vaues are in (Tage seit letzter kampagne).
# sns.heatmap(train_data.isnull(),yticklabels=False)
# plt.show()
#
# Looking at the data, it seems that where we have a NaN in [Tage...] the value in [Anzahl...] is 0 else we have non-zero variable.
# Meanwhile, instead of droping this column [Tage...] I will fill the missing values by 0.
train_data=train_data.fillna(0)
test_data=test_data.fillna(0)
#
#
#
#
#
#
# I change the continuse variables to categorical. For Kontostand I have 3 (negative: 1, positive:2 and 0: 0). And for [Tage...] I have 0 and positive:1.
train_data['Kontostand'] = ['Negative' if x < 0 else 'Null' if x == 0 else 'positive' if x > 0 else 'NaN' for x in train_data['Kontostand']]
train_data['Tage_seit_letzter_Kampagne'] = [1 if x > 0 else 0 for x in train_data['Tage_seit_letzter_Kampagne']]

test_data['Kontostand'] = ['Negative' if x < 0 else 'Null' if x == 0 else 'positive' if x > 0 else 'NaN' for x in test_data['Kontostand']]
test_data['Tage_seit_letzter_Kampagne'] = [1 if x > 0 else 0 for x in test_data['Tage_seit_letzter_Kampagne']]

#
cleanup_nums = {'Zielvariable':{'nein':0,'ja':1}}
train_data.replace(cleanup_nums, inplace=True)

cat_vars=['Kontostand','Geschlecht','Familienstand','Haus','Kredit','Ausfall_Kredit','Kontaktart','Schulabschluß',
                'Ergebnis_letzte_Kampagne','Art_der_Anstellung','Monat']
for var in cat_vars:
    cat_list_train='var'+'_'+ var
    cat_list_train = pd.get_dummies(train_data[var], prefix=var)
    cat_list_test= 'var' + '_' + var
    cat_list_test = pd.get_dummies(test_data[var], prefix=var)
    data1=train_data.join(cat_list_train)
    data2=test_data.join(cat_list_test)
    train_data=data1
    test_data=data2

data_vars_train=train_data.columns.values.tolist()
data_vars_test=test_data.columns.values.tolist()
to_keep_train=[i for i in data_vars_train if i not in cat_vars]
Train_dt=train_data[to_keep_train]
to_keep_test=[i for i in data_vars_test if i not in cat_vars]
Test_dt=test_data[to_keep_test]
print(Train_dt.columns)
print(Test_dt.columns)
#


# droping the unuseful columns in modeling.
Train_dt=Train_dt.drop(['Tag','Stammnummer','Anruf-ID'],axis=1)
Test_dt=Test_dt.drop(['Tag','Stammnummer','Anruf-ID','Zielvariable'],axis=1)


#Data Normalization
scaler=preprocessing.StandardScaler()
Train_dt['normAlter'] = scaler.fit_transform(Train_dt[['Alter']])
Train_dt['normDauer'] = scaler.fit_transform(Train_dt[['Dauer']])
Test_dt['normAlter'] = scaler.fit_transform(Test_dt[['Alter']])
Test_dt['normDauer'] = scaler.fit_transform(Test_dt[['Dauer']])
Test_dt=Test_dt.drop(['Alter','Dauer'], axis=1)
Tain_dt = Train_dt.drop(['Alter','Dauer'], axis=1)

# For better convergance let us change float data type to int64.
def change_type(Train_dt):
    float_list = list(Train_dt.select_dtypes(include=["float64"]).columns)
    print(float_list)
    for col in float_list:
        Train_dt[col] = Train_dt[col].astype(np.int64)


change_type(Train_dt)
change_type(Test_dt)
#
#
#
#
#
# # # ///////// Data Balancing ///////////////////

#The data where imbalanced. To do the oversampling I use the SMOTE method
#

features_name=['Anzahl_der_Ansprachen',
       'Tage_seit_letzter_Kampagne', 'Anzahl_Kontakte_letzte_Kampagne',
       'Kontostand_Negative', 'Kontostand_Null', 'Kontostand_positive',
       'Geschlecht_m', 'Geschlecht_w', 'Familienstand_geschieden',
       'Familienstand_single', 'Familienstand_verheiratet', 'Haus_ja',
       'Haus_nein', 'Kredit_ja', 'Kredit_nein', 'Ausfall_Kredit_ja',
       'Ausfall_Kredit_nein', 'Kontaktart_Festnetz', 'Kontaktart_Handy',
       'Kontaktart_Unbekannt', 'Schulabschluß_Abitur',
       'Schulabschluß_Real-/Hauptschule', 'Schulabschluß_Studium',
       'Schulabschluß_Unbekannt', 'Ergebnis_letzte_Kampagne_Erfolg',
       'Ergebnis_letzte_Kampagne_Kein Erfolg',
       'Ergebnis_letzte_Kampagne_Sonstiges',
       'Ergebnis_letzte_Kampagne_Unbekannt', 'Art_der_Anstellung_Arbeiter',
       'Art_der_Anstellung_Arbeitslos', 'Art_der_Anstellung_Dienstleistung',
       'Art_der_Anstellung_Gründer', 'Art_der_Anstellung_Hausfrau',
       'Art_der_Anstellung_Management', 'Art_der_Anstellung_Rentner',
       'Art_der_Anstellung_Selbständig', 'Art_der_Anstellung_Student',
       'Art_der_Anstellung_Technischer Beruf', 'Art_der_Anstellung_Unbekannt',
       'Art_der_Anstellung_Verwaltung', 'Monat_apr', 'Monat_aug', 'Monat_dec',
       'Monat_feb', 'Monat_jan', 'Monat_jul', 'Monat_jun', 'Monat_mar',
       'Monat_may', 'Monat_nov', 'Monat_oct', 'Monat_sep', 'normAlter',
       'normDauer']
#
X=Train_dt[['Anzahl_der_Ansprachen',
       'Tage_seit_letzter_Kampagne', 'Anzahl_Kontakte_letzte_Kampagne',
       'Kontostand_Negative', 'Kontostand_Null', 'Kontostand_positive',
       'Geschlecht_m', 'Geschlecht_w', 'Familienstand_geschieden',
       'Familienstand_single', 'Familienstand_verheiratet', 'Haus_ja',
       'Haus_nein', 'Kredit_ja', 'Kredit_nein', 'Ausfall_Kredit_ja',
       'Ausfall_Kredit_nein', 'Kontaktart_Festnetz', 'Kontaktart_Handy',
       'Kontaktart_Unbekannt', 'Schulabschluß_Abitur',
       'Schulabschluß_Real-/Hauptschule', 'Schulabschluß_Studium',
       'Schulabschluß_Unbekannt', 'Ergebnis_letzte_Kampagne_Erfolg',
       'Ergebnis_letzte_Kampagne_Kein Erfolg',
       'Ergebnis_letzte_Kampagne_Sonstiges',
       'Ergebnis_letzte_Kampagne_Unbekannt', 'Art_der_Anstellung_Arbeiter',
       'Art_der_Anstellung_Arbeitslos', 'Art_der_Anstellung_Dienstleistung',
       'Art_der_Anstellung_Gründer', 'Art_der_Anstellung_Hausfrau',
       'Art_der_Anstellung_Management', 'Art_der_Anstellung_Rentner',
       'Art_der_Anstellung_Selbständig', 'Art_der_Anstellung_Student',
       'Art_der_Anstellung_Technischer Beruf', 'Art_der_Anstellung_Unbekannt',
       'Art_der_Anstellung_Verwaltung', 'Monat_apr', 'Monat_aug', 'Monat_dec',
       'Monat_feb', 'Monat_jan', 'Monat_jul', 'Monat_jun', 'Monat_mar',
       'Monat_may', 'Monat_nov', 'Monat_oct', 'Monat_sep', 'normAlter',
       'normDauer']]
y=Train_dt[['Zielvariable']]
# #
#
# #
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
os = SMOTE(random_state=0)
columns = x_train.columns
os_data_X,os_data_y = os.fit_sample(x_train, y_train)
os_data_X = pd.DataFrame(data=os_data_X,columns=columns )
os_data_y= pd.DataFrame(data=os_data_y,columns=['Zielvaribal'])
print("length of oversampled data is ",len(os_data_X))
print("Number of no subscription in oversampled data",len(os_data_y[os_data_y['Zielvaribal']==0]))
print("Number of subscription",len(os_data_y[os_data_y['Zielvaribal']==1]))
print("Proportion of no subscription data in oversampled data is ",len(os_data_y[os_data_y['Zielvaribal']==0])/len(os_data_X))
print("Proportion of subscription data in oversampled data is ",len(os_data_y[os_data_y['Zielvaribal']==1])/len(os_data_X))

#
# ///////////////////Machine Learning Implemention/////////////////////////////////////
# # //////////////////Features Selection For Logistic Regression//////////////////

from sklearn.model_selection import cross_val_score

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier

#
data_final_vars=Train_dt.columns.values.tolist()
y=os_data_y['Zielvaribal']
X=[i for i in data_final_vars if i not in y]

# feature extraction
model = ExtraTreesClassifier()
fit = model.fit(os_data_X, os_data_y.values.ravel())


for feature in zip(features_name, fit.feature_importances_):
    print(feature)

from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score

sfm = SelectFromModel(model, threshold=0.01)

# Train the selector
sfm.fit(os_data_X, os_data_y.values.ravel())
for feature_list_index in sfm.get_support(indices=True):
    print(features_name[feature_list_index])

Et_selected=['Anzahl_der_Ansprachen','Geschlecht_m','Geschlecht_w','Familienstand_geschieden','Familienstand_single','Familienstand_verheiratet'
,'Haus_ja','Haus_nein','Kredit_ja','Kredit_nein','Kontaktart_Handy','Kontaktart_Unbekannt','Schulabschluß_Abitur','Schulabschluß_Real-/Hauptschule','Schulabschluß_Studium',
'Art_der_Anstellung_Arbeiter','Art_der_Anstellung_Management','Art_der_Anstellung_Technischer Beruf','Art_der_Anstellung_Verwaltung','Monat_aug','Monat_jul','Monat_jun','Monat_may',
             'normAlter','normDauer']





# data_final_vars=Train_dt.columns.values.tolist()
# y=os_data_y['Zielvaribal']
# X=[i for i in data_final_vars if i not in y]
# # feature extraction
#
# model = LogisticRegression()
# rfe = RFE(model, 20)
# fit = rfe.fit(os_data_X, os_data_y.values.ravel())
#
# for feature in zip(features_name, fit.ranking_):
#      print(feature)
# #
# #
# selected=['Kontostand_positive','Geschlecht_m','Geschlecht_w','Familienstand_geschieden','Familienstand_single','Familienstand_verheiratet','Haus_ja','Haus_nein',
# 'Ausfall_Kredit_nein','Schulabschluß_Abitur','Schulabschluß_Real-/Hauptschule','Schulabschluß_Studium','Art_der_Anstellung_Arbeiter','Art_der_Anstellung_Management'
# ,'Art_der_Anstellung_Technischer Beruf','Art_der_Anstellung_Verwaltung','Monat_aug','Monat_jul','Monat_jun','Monat_may']
selected_x=os_data_X[Et_selected]
y=os_data_y['Zielvaribal']
logit_model=sm.Logit(y,selected_x)
result=logit_model.fit()
print(result.summary2())


opt_Et_selected=['Anzahl_der_Ansprachen','Geschlecht_m','Geschlecht_w','Familienstand_geschieden','Familienstand_single','Familienstand_verheiratet'
,'Haus_ja','Haus_nein','Kredit_nein','Kontaktart_Handy','Kontaktart_Unbekannt','Schulabschluß_Abitur','Schulabschluß_Real-/Hauptschule',
'Art_der_Anstellung_Arbeiter','Art_der_Anstellung_Management','Art_der_Anstellung_Technischer Beruf','Art_der_Anstellung_Verwaltung','Monat_aug','Monat_jul','Monat_jun','Monat_may',
             'normAlter','normDauer']


selected_x=os_data_X[opt_Et_selected]
y=os_data_y['Zielvaribal']
logit_model=sm.Logit(y,selected_x)
result=logit_model.fit()
print(result.summary2())


from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import GridSearchCV

# penalty = ['l1', 'l2']
#
# # Create regularization hyperparameter space
# C = np.logspace(0, 4, 10)
#
# # Create hyperparameter options
# hyperparameters = dict(C=C, penalty=penalty)


# X_train, X_test, y_train, y_test = train_test_split(selected_X, y, test_size=0.3, random_state=0)
# Et_selected=['Anzahl_der_Ansprachen','Geschlecht_m','Geschlecht_w','Familienstand_geschieden','Familienstand_single','Familienstand_verheiratet'
# ,'Haus_ja','Haus_nein','Kredit_ja','Kredit_nein','Kontaktart_Handy','Kontaktart_Unbekannt','Schulabschluß_Abitur','Schulabschluß_Real-/Hauptschule','Schulabschluß_Studium',
# 'Art_der_Anstellung_Arbeiter','Art_der_Anstellung_Management','Art_der_Anstellung_Technischer Beruf','Art_der_Anstellung_Verwaltung','Monat_aug','Monat_jul','Monat_jun','Monat_may',
#              'normAlter','normDauer']

selected_x=os_data_X[opt_Et_selected]
X_train, X_test, y_train, y_test = train_test_split(selected_x, y, test_size=0.3, random_state=0)
logreg = LogisticRegression()
# clf = GridSearchCV(logreg, hyperparameters, cv=5, verbose=0)
# best_model = clf.fit(X_train, y_train)
# y_pred_best= best_model.predict(X_test)
# print('Best Penalty:', best_model.best_estimator_.get_params()['penalty'])
# print('Best C:', best_model.best_estimator_.get_params()['C'])
# print('Accuracy of logistic regression (GridSearchCV) classifier on test set: {:.2f}'.format(logreg.score(y_test, y_pred_best)))

predicted = cross_val_predict(logreg, selected_x, y, cv=10)
logreg.fit(X_train, y_train)
#
y_pred_lr= logreg.predict(X_test)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test,y_test)))
print('Logistic-regression (cross_validation):',metrics.accuracy_score(y, predicted))
#
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_pred_lr,y_test)
print(confusion_matrix)
# #
from sklearn.metrics import classification_report
print(classification_report(y_pred_lr,y_test))
# #
# # # ////////////////////////////////////Decision Tree Classifier////////////////////////////////////////////////
#
#
dtc = DecisionTreeClassifier()
dtc.fit(X_train, y_train)

y_pred = dtc.predict(X_test)
print('Accuracy of Decision Tree classifier on test set: {:.2f}'.format(dtc.score(X_test, y_test)))
#
#
#////////////////////////Feature SElECTION for RANDOM FOREST///////////////////////////////////////////////////
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier


# data_final_vars=Train_dt.columns.values.tolist()
# y=os_data_y['Zielvaribal']
# X=[i for i in data_final_vars if i not in y]
# # feature extraction
#
# model = RandomForestClassifier()
# fit = model.fit(os_data_X, os_data_y.values.ravel())
# for feature in zip(features_name, fit.feature_importances_):
#     print(feature)
#
#
# from sklearn.feature_selection import SelectFromModel
# from sklearn.metrics import accuracy_score
#
# sfm = SelectFromModel(model, threshold=0.02)
#
# # Train the selector
# sfm.fit(os_data_X, os_data_y.values.ravel())
# for feature_list_index in sfm.get_support(indices=True):
#     print(features_name[feature_list_index])
#
#
#
# rf_selected=['Anzahl_der_Ansprachen','Geschlecht_m','Geschlecht_w','Familienstand_verheiratet','Haus_ja','Kredit_ja','Kontaktart_Handy','Kontaktart_Unbekannt'
#              ,'Schulabschluß_Abitur','Art_der_Anstellung_Arbeiter','Art_der_Anstellung_Technischer Beruf','Monat_aug','Monat_jul','Monat_may','normAlter','normDauer']
# selected_X=os_data_X[rf_selected]
#
# X_train, X_test, y_train, y_test = train_test_split(selected_X, y, test_size=0.3, random_state=0)
rf = RandomForestClassifier()
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)
print('Accuracy of Random Forest classifier on test set: {:.2f}'.format(rf.score(X_test, y_test)))
#
# # ////////////////////////////////////AUC//////////////////////////////////////
from sklearn import metrics
import matplotlib.pyplot as plt

plt.figure()

models = [
    {
        'label': 'Logistic Regression',
        'model': LogisticRegression(),
    },
    {
        'label': 'Logistic Regression, C=0.001',
        'model':  LogisticRegression(penalty='l2', C=0.001, random_state=0)
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
        'label': 'Logistic Regression, C=1',
        'model':  LogisticRegression(penalty='l2', C=1, random_state=0)
    },
    {
        'label': 'Random Forest',
        'model': RandomForestClassifier(),
    },
    {
        'label': 'Decision Tree',
        'model': DecisionTreeClassifier(),
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

# # /////////////////////////////////////WINNER MODEL//////////////////////////////////
# selected_X=os_data_X[Et_selected]
#
# X_train, X_test, y_train, y_test = train_test_split(selected_X, y, test_size=0.3, random_state=0)
winner_model=RandomForestClassifier()

winner_model.fit(X_train, y_train)

Test_Data=Test_dt[opt_Et_selected]

# Test_Data=Test_dt[['Anzahl_der_Ansprachen','Geschlecht_m','Geschlecht_w','Familienstand_geschieden','Familienstand_single','Familienstand_verheiratet'
# ,'Haus_ja','Haus_nein','Kredit_ja','Kredit_nein','Kontaktart_Handy','Kontaktart_Unbekannt','Schulabschluß_Abitur','Schulabschluß_Real-/Hauptschule','Schulabschluß_Studium',
# 'Art_der_Anstellung_Arbeiter','Art_der_Anstellung_Management','Art_der_Anstellung_Technischer Beruf','Art_der_Anstellung_Verwaltung','Monat_aug','Monat_jul','Monat_jun','Monat_may',
#              'normAlter','normDauer']]
#
prediction=pd.DataFrame(winner_model.predict(Test_Data),columns=['Zielvariable'])
proba=pd.DataFrame(list(winner_model.predict_proba(Test_Data)),columns=['zero','one'])
print(proba.head())
#
#
#
test_Id=test_data['Stammnummer']
predicted=proba['one']

my_submission = pd.DataFrame({'ID': test_Id, 'Expected': predicted})

my_submission.to_csv('submission.csv', index=False)
result=pd.read_csv('submission.csv')
print(result.head())