# Features Selecetion
data_final_vars=Train_dt.columns.values.tolist()
y = smt_data_y['Zielvariable']
X = [i for i in data_final_vars if i not in y]

# feature extraction
model = ExtraTreesClassifier()
fit = model.fit(smt_data_X, smt_data_y.values.ravel())
for feature in zip(Features_name, fit.feature_importances_):
    print(feature)


sfm = SelectFromModel(model, threshold=0.01)
sfm.fit(smt_data_X, smt_data_y.values.ravel())
Features_name_list=[]
for feature_list_index in sfm.get_support(indices=True):
    Features_name_list.append(Features_name[feature_list_index])
    print(Features_name[feature_list_index])


# create summary to get some information about the statistics
Et_selected=Features_name_list
selected_x=smt_data_X[Et_selected]
y=smt_data_y['Zielvariable']
logit_model=sm.Logit(y,selected_x)
result=logit_model.fit()
print(result.summary2())


# removing the features with p_values>0.05
unwanted={'Alter','Geschlecht_m','Kontostand_Negative','Kontostand_Null','Kontostand_positive','Geschlecht_w',
          'Familienstand_geschieden','Art_der_Anstellung_Dienstleistung','Art_der_Anstellung_Technischer Beruf','Art_der_Anstellung_Management',
        'Art_der_Anstellung_Verwaltung','Familienstand_single','Familienstand_verheiratet','Monat_aug','Monat_jul','Haus_ja','Haus_nein','Kredit_ja',
          'Kredit_nein','Ergebnis_letzte_Kampagne_Unbekannt'}


opt_Et_selected=[elm for elm in Et_selected if elm not in unwanted]

#creat new summary to check the p-values
selected_x=smt_data_X[opt_Et_selected]
y=smt_data_y['Zielvariable']
logit_model=sm.Logit(y,selected_x)
result=logit_model.fit()
print(result.summary2())