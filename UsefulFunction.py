train_data.rename(columns={ 'Schulabschlu�' : 'Schulabschluss'}, inplace=True)
test_data.rename(columns={ 'Schulabschlu�' : 'Schulabschluss'}, inplace=True)

def change_string(x):
    return x.replace('Gr�nder',"Greunder").replace('Selbst�ndig','Selbsteandig')

train_data['Art_der_Anstellung'] = train_data['Art_der_Anstellung'].map(lambda x: change_string(x))
test_data['Art_der_Anstellung'] = test_data['Art_der_Anstellung'].map(lambda x: change_string(x))

