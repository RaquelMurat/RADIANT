from numpy.random import seed
seed(1)
import tensorflow as tf
tf.random.set_seed(1)
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, Callback, ReduceLROnPlateau
from tqdm import *
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import shuffle
from transformers import BertTokenizer, TFBertModel , TFBertForSequenceClassification, AutoTokenizer, AutoModel,
                         XLMForSequenceClassification, XLMTokenizer, TFXLMForSequenceClassification
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import AUC
import os
import json
import random
import tensorflow.keras as keras
import shutil
import warnings
from sklearn.metrics import roc_auc_score
from MultipleClassAUROC import MultipleClassAUROC as MultipleClassAUROC
import sys
from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from hyperas.distributions import choice, uniform


random.seed(1234)   # para la division en train y test y la inicializacion de pesos

# **** RECOGER PARAMETROS Y CREAR FICHEROS ****
modelo = sys.argv[1] 
idioma = sys.argv[2] 
print("Modelo ", modelo)
print("Idioma ", idioma)

if modelo not in ["beto", "bert", "bioBert", "xlm"]:
    print("El nombre del modelo es incorrecto.")
    exit(-1)

if idioma not in ["en", "esp", "indio"]:
    print("El idioma es incorrecto.")
    exit(-1)
    
# Creo el fichero donde se guardaran los pesos de la red entrenada
filepath = "./checkpoints_padchest/reduceOnPlateau/" + modelo + "_" + idioma + "_reduce.h5"
print(filepath)
f = open(filepath, "w+")
f.close()


# ***** CARGAR EL DATASET ****

data = pd.read_csv("/scratch/codigofsoler/baseDeDatos/csvs/5ZonascustomPADCHEST_onehot_multicolumn_SOLO_KATTY.csv")
data = data[["Report", "Pulmon", "Calcificacion", "Cuerpos extranos", "Mediastino e hilios pulmonares", 
             "Pleura y diafragma y pared abdominal", "Patologica", "Unchanged"]]    # me quedo solo con las columnas que me interesan

if idioma == "en":
    informes = list(json.load(open("./informes_padchest_en.json")))
    data["Report"] = informes
elif idioma == "esp":
    informes = list(json.load(open("./informes_padchest_esp.json")))
    data["Report"] = informes



# Divido en train, validation y test
new_train, new_test = train_test_split(data, test_size = 0.2, random_state = 1, shuffle = True, stratify = data[['Patologica']])    

new_train, new_val =  train_test_split(new_train, test_size = 0.1, random_state = 1, shuffle = True, stratify = new_train[['Patologica']])

# Guardo los informes de entrada
train_comment = new_train["Report"].values
test_comment  = new_test["Report"].values
val_comment = new_val["Report"].values
        
        
# ******************** CARGAR TOKENIZER Y FORMATEAR INFORMES DE ENTRADA ********************

if modelo == "beto":
    model_name = 'dccuchile/bert-base-spanish-wwm-cased'
    tokenizer = BertTokenizer.from_pretrained(pretrained_model_name_or_path = model_name) 

elif modelo == "bertmulti":
    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
elif modelo == "bert":
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
elif modelo == "bioBert":
    tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
elif modelo == "xlm":
    tokenizer = XLMTokenizer.from_pretrained('xlm-mlm-en-2048')


padded_ids_train = []
mask_ids_train = []

# 4 opciones para la maxima longitud de reporte
max_length = {{choice([60, 118, 134, 154])}}

for i in tqdm(range(len(train_comment))):
    encoding = tokenizer.encode_plus(train_comment[i]  , max_length = max_length , pad_to_max_length = True, truncation = "longest_first")
    input_ids , attention_id = encoding["input_ids"] , encoding["attention_mask"] 
    padded_ids_train.append(input_ids)
    mask_ids_train.append(attention_id)
    
padded_ids_test = []
mask_ids_test = []
for i in tqdm(range(len(test_comment))):
    encoding = tokenizer.encode_plus(test_comment[i]  , max_length = max_length , pad_to_max_length = True , truncation = "longest_first" )
    input_ids , attention_id = encoding["input_ids"] , encoding["attention_mask"]
    padded_ids_test.append(input_ids)
    mask_ids_test.append(attention_id)

padded_ids_val = []
mask_ids_val = []
for i in tqdm(range(len(val_comment))):
    encoding = tokenizer.encode_plus(val_comment[i]  , max_length = max_length , pad_to_max_length = True , truncation = "longest_first" )
    input_ids , attention_id = encoding["input_ids"] , encoding["attention_mask"]
    padded_ids_val.append(input_ids)
    mask_ids_val.append(attention_id)

y_train = new_train.drop(["Report"] , axis=1)   # En y_train se guardan los 1 y 0 de cada columna para luego usarlos en evaluate
train_id = np.array(padded_ids_train)           # train_id y train_test son los datos de entrada para entrenar (evaluar para test y val)
train_mask = np.array(mask_ids_train)

y_test = new_test.drop(["Report"] , axis=1)     # Analogo a y_train
test_id = np.array(padded_ids_test)
test_mask = np.array(mask_ids_test)

y_val = new_val.drop(["Report"] , axis=1)       # Analogo a y_train
val_id = np.array(padded_ids_val)
val_mask = np.array(mask_ids_val)


validation_data = ([val_id , val_mask], y_val)  # Usara estos datos para calcular val_auc de cada epoca de entrenamiento

# *************** ARQUITECTURA DEL MODELO ****************

input_1 = tf.keras.Input(shape = (max_length) , dtype=np.int32)    # Recibe train_id
input_2 = tf.keras.Input(shape = (max_length) , dtype=np.int32)    # Recibe train_mask

# Cargo el modelo pedido
if modelo == "beto":
    model = TFBertForSequenceClassification.from_pretrained("/home/murat/datasets/pytorch", from_pt=True)
elif modelo == "bertmulti":
    model = TFBertForSequenceClassification.from_pretrained('bert-base-multilingual-cased')
elif modelo == "bert":
    model = TFBertForSequenceClassification.from_pretrained('bert-base-cased')
elif modelo == "bioBert":
    model = TFBertForSequenceClassification.from_pretrained("emilyalsentzer/Bio_ClinicalBERT", from_pt=True)
elif modelo == "xlm":
    model = TFXLMForSequenceClassification.from_pretrained('xlm-mlm-en-2048', from_pt=True)

output  = model([input_1 , input_2] , training = True)
answer = tf.keras.layers.Dense(7 , activation = tf.nn.sigmoid )(output[0])  # Capa densa con 7 salidas (pulmon, ..., unchanged)
model = tf.keras.Model(inputs = [input_1, input_2 ] , outputs = [answer])   # Construye la arquitectura sobre el modelo
model.summary()

#model.load_weights("./checkpoints_padchest/best_xlm100_en.h5")


# ********* CALLBACKS, CHECKPOINTS, CLASSWEIGHTS *****************

# Cargo el diccionario de frecuencias para calcular los class weights.
#Para cada clase, su class_weight sera la inversa del numero de apariciones
d_frecuencias = json.load(open("/scratch/codigofsoler/baseDeDatos/diccionarios/d_frecuencias_5zonas_sin_diagnosticos.json"))
class_weights = {}
nsamples = len(data)
nclasses = 7
class_weights[0] = (nsamples*1.0)/(d_frecuencias["Pulmon"]*nclasses)
class_weights[1] = (nsamples*1.0)/(d_frecuencias["Calcificacion"]*nclasses)
class_weights[2] = (nsamples*1.0)/(d_frecuencias["Cuerpos extranos"]*nclasses)
class_weights[3] = (nsamples*1.0)/(d_frecuencias["Mediastino e hilios pulmonares"]*nclasses)
class_weights[4] = (nsamples*1.0)/(d_frecuencias["Pleura y diafragma y pared abdominal"]*nclasses)
class_weights[5] = (nsamples*1.0)/(d_frecuencias["Patologica"]*nclasses)
class_weights[6] = (nsamples*1.0)/(d_frecuencias["Unchanged"]*nclasses)


# Learning rate scheduler. El reduceOnPlateau reduce el lr segun factor cuando hayana pasado patience epocas sin mejora en el auc
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                              patience=2, min_lr=3e-8, verbose = 1)


# El checkpoint guarda en filepath los pesos de la red en la mejor epoca del entrenamiento (mejor monitor)
checkpoint = ModelCheckpoint(filepath=filepath,
                                  monitor='val_auc',
                                  verbose=1,
                                  save_best_only=True , mode = "max" , save_weights_only = True)

# Este callback es para mostrar el val_auc por clase al final de cada epoca
nombres_clases = ['Pulmon', 'Calcificacion', 'Cuerpos extranos', 'Mediastino e hilios pulmonares', 'Pleura y diafragma y pared abdominal', 'Patologica', 'Unchanged']
auroc = MultipleClassAUROC(class_names = nombres_clases, sequence = validation_data, weights_path = filepath)




# ************* COMPILAR Y ENTRENAR ************************
auc_score = AUC(multi_label=True)   # Metrica

# 3 opciones para optimizador, cada una con 3 lr inciales
adam    = keras.optimizers.Adam(lr={{choice([3e-4, 3e-5, 3e-6])}})
rmsprop = keras.optimizers.RMSprop(lr={{choice([3e-4, 3e-5, 3e-6])}})
sgd     = keras.optimizers.SGD(lr={{choice([3e-4, 3e-5, 3e-6])}})
optim = {"adam": adam, "rmsprop": rmsprop, 
         "sgd": sgd}[{{choice(['adam', 'sgd', 'rmsprop'])}}]

model.compile(optimizer = optim,
                loss = tf.keras.losses.binary_crossentropy,
                metrics = [auc_score]
)

# 4 opciones para batch_size
model.fit(x = [train_id , train_mask] , y = y_train,
          validation_data = validation_data , 
          batch_size={{choice([32, 64, 128, 256])}}, 
          epochs=12, callbacks = [checkpoint, reduce_lr, auroc,], class_weight = class_weights
)

# ****************** MODEL PREDICTION *****************
print("\n\n\n")
print("RESULTADOS USANDO " , modelo, " CON ", idioma, "Y lrReduceOnPlateau")
print("\n")

model.evaluate([test_id , test_mask] , y_test)  # Evalua usando los datos de test (no los habia visto nunca)
y_hat = model.predict([test_id , test_mask])    # Guarda las predicciones de test
y = np.array(y_test)
for i in range(len(list(data))-1):              # Para ver el auc final por clase
    print(list(data)[i+1], ": ", roc_auc_score(y_true = y[:, i], y_score = y_hat[:, i]))

sub = pd.DataFrame(y_hat , columns=nombres_clases)
sub["Report"] = test_comment    # Le anyado los informes

sub = sub[["Report", 'Pulmon', 'Calcificacion', 'Cuerpos extranos', 'Mediastino e hilios pulmonares', 
           'Pleura y diafragma y pared abdominal', 'Patologica', 'Unchanged']]  # Pongo los informes delante para verlos mejor



print(sub.head())   # Comparo los 10 primeros datos reales con las 10 primeras predicciones
print(new_test.head())

print("\n\n\n\n")

print(sub.tail())   # Comparo los 10 ultimos datos reales con las 10 ultimas predicciones
print(new_test.tail())




exit()





















