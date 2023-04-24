import pandas as pd
import numpy as np
import pickle
import seaborn as sns
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report

#vamos a probar el modelo elegido para que nos mida el grado de satisfacción del cliente a traves del predict_proba
#con el modelo random forest entrenado



def satisfaction():
    random_forest_model = pickle.load(open('./models/random_forest.pkl', 'rb'))
    data_prueba = pd.read_csv('./data/processed/df_test.csv').sample(frac=1)
    respuesta = data_prueba['satisfaction'].iloc[:1].min()
    print(f'respuesta encuesta: {respuesta}')
    pred_proba = random_forest_model.predict_proba(data_prueba.iloc[:1:, :18:].values)
    #print(pred_proba)
    grado_sast = list(pred_proba[0])
    #print(grado_sast[0])


    if respuesta == 0:
            print('grado de no satisfacción es del: {:.0f}%'.format(grado_sast[0]*100))
    else:
        print('grado de satisfacción es del: {:.0f}%'.format(grado_sast[0]*100))


satisfaction()