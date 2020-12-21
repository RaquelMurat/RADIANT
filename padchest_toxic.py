from numpy.random import seed
seed(1)
import tensorflow as tf
tf.random.set_seed(1)
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, Callback
from tqdm import *
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import shuffle
from transformers import BertTokenizer, TFBertModel , TFBertForSequenceClassification, AutoTokenizer, AutoModel , XLMForSequenceClassification, XLMTokenizer, TFXLMForSequenceClassification
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import AUC
import os
# from googletrans import *
# from wordcloud import WordCloud , STOPWORDS
import json
import random
import shutil
import warnings
from sklearn.metrics import roc_auc_score

# ***** LOADING THE DATA ****

random.seed(1234)   # para la division en train y test y la inicializacion de pesos
data = pd.read_csv("/scratch/codigofsoler/baseDeDatos/csvs/5ZonascustomPADCHEST_onehot_multicolumn_SOLO_KATTY.csv")
data = data[["Report", "Pulmon", "Calcificacion", "Cuerpos extranos", "Mediastino e hilios pulmonares", "Pleura y diafragma y pared abdominal", "Patologica", "Unchanged"]]

informes = list(json.load(open("./informes_padchest_en.json")))
data["Report"] = informes

new_train, new_test = train_test_split(data, test_size = 0.2, random_state = 1, shuffle = True, stratify = data[['Patologica']])


train_comment = new_train["Report"].values
test_comment  = new_test["Report"].values

        
        
# ******************** Loading the bert tokenizer and encoding the text in input format********************

#model_name = 'dccuchile/bert-base-spanish-wwm-cased'
#tokenizer = BertTokenizer.from_pretrained(pretrained_model_name_or_path = model_name) 

#tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')

#tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

#tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")

tokenizer = XLMTokenizer.from_pretrained('xlm-mlm-17-1280')



padded_ids_train = []
mask_ids_train = []
for i in tqdm(range(len(train_comment))):
    encoding = tokenizer.encode_plus(train_comment[i]  , max_length = 128 , pad_to_max_length = True, truncation = "longest_first")
    input_ids , attention_id = encoding["input_ids"] , encoding["attention_mask"] 
    padded_ids_train.append(input_ids)
    mask_ids_train.append(attention_id)
    
    


padded_ids_test = []
mask_ids_test = []

for i in tqdm(range(len(test_comment))):
    encoding = tokenizer.encode_plus(test_comment[i]  , max_length = 128 , pad_to_max_length = True , truncation = "longest_first" )
    input_ids , attention_id = encoding["input_ids"] , encoding["attention_mask"]
    padded_ids_test.append(input_ids)
    mask_ids_test.append(attention_id)

    
    

y_train = new_train.drop(["Report"] , axis=1)
y_test = new_test.drop(["Report"] , axis=1)


train_id = np.array(padded_ids_train)
train_mask = np.array(mask_ids_train)

test_id = np.array(padded_ids_test)
test_mask = np.array(mask_ids_test)



# *************** ARQUITECTURA ****************

input_1 = tf.keras.Input(shape = (128) , dtype=np.int32)
input_2 = tf.keras.Input(shape = (128) , dtype=np.int32)


#model = TFBertForSequenceClassification.from_pretrained("/home/murat/datasets/pytorch", from_pt=True)
#model = TFBertForSequenceClassification.from_pretrained('bert-base-multilingual-cased')
#model = TFBertForSequenceClassification.from_pretrained('bert-base-cased')
#model = TFBertForSequenceClassification.from_pretrained("emilyalsentzer/Bio_ClinicalBERT", from_pt=True)
model = TFXLMForSequenceClassification.from_pretrained('xlm-mlm-17-1280', from_pt=True)

output  = model([input_1 , input_2] , training = True )
answer = tf.keras.layers.Dense(7 , activation = tf.nn.sigmoid )(output[0])
model = tf.keras.Model(inputs = [input_1, input_2 ] , outputs = [answer])
model.summary()


#model.load_weights("./checkpoints_padchest/xlm17_en_semilla1.h5")



# ********* OPTIMIZADOR , CHECKPOINTS Y CLASSWEIGHTS *****************
d_frecuencias = json.load(open("/scratch/codigofsoler/baseDeDatos/diccionarios/d_frecuencias_5zonas_sin_diagnosticos.json"))


nsamples = len(data)
nclasses = 7

medias = [d_frecuencias["Pulmon"]/nsamples,
          d_frecuencias["Calcificacion"]/nsamples,
          d_frecuencias["Cuerpos extranos"]/nsamples,
          d_frecuencias["Mediastino e hilios pulmonares"]/nsamples,
          d_frecuencias["Pleura y diafragma y pared abdominal"]/nsamples,
          d_frecuencias["Patologica"]/nsamples,
          d_frecuencias["Unchanged"]/nsamples,
          ]


class_weights = {}


class_weights[0] = (nsamples*1.0)/(d_frecuencias["Pulmon"]*nclasses)
class_weights[1] = (nsamples*1.0)/(d_frecuencias["Calcificacion"]*nclasses)
class_weights[2] = (nsamples*1.0)/(d_frecuencias["Cuerpos extranos"]*nclasses)
class_weights[3] = (nsamples*1.0)/(d_frecuencias["Mediastino e hilios pulmonares"]*nclasses)
class_weights[4] = (nsamples*1.0)/(d_frecuencias["Pleura y diafragma y pared abdominal"]*nclasses)
class_weights[5] = (nsamples*1.0)/(d_frecuencias["Patologica"]*nclasses)
class_weights[6] = (nsamples*1.0)/(d_frecuencias["Unchanged"]*nclasses)





filepath = "./checkpoints_padchest/xlm17_en_semilla1.h5.h5"

checkpoint = ModelCheckpoint(filepath=filepath,
                                  monitor='val_auc',
                                  verbose=1,
                                  save_best_only=True , mode = "max" , save_weights_only = True)

auc_score = AUC(multi_label=True)

auroc = MultipleClassAUROC(class_names = ['Pulmon', 'Calcificacion', 'Cuerpos extranos', 'Mediastino e hilios pulmonares', 'Pleura y diafragma y pared abdominal', 'Patologica', 'Unchanged'])



# ************* COMPILAR Y ENTRENAR ************************
model.compile(optimizer = Adam(lr = 3e-5),
                loss = tf.keras.losses.binary_crossentropy,
                metrics = [auc_score]
)



model.fit([train_id , train_mask] , y_train,
          validation_split = 0.1 , batch_size = 32, 
          epochs=4, callbacks = [checkpoint], class_weight = class_weights
)

# ****************** MODEL PREDICTION *****************

model.save_weights(filepath)
model.evaluate([test_id , test_mask] , y_test)
a = model.predict([test_id , test_mask])
sub = pd.DataFrame(a , columns=['Pulmon', 'Calcificacion', 'Cuerpos extranos', 'Mediastino e hilios pulmonares', 'Pleura y diafragma y pared abdominal', 'Patologica', 'Unchanged'])
sub["Report"] = test_comment

sub = sub[["Report", 'Pulmon', 'Calcificacion', 'Cuerpos extranos', 'Mediastino e hilios pulmonares', 'Pleura y diafragma y pared abdominal', 'Patologica', 'Unchanged']]

# XLM EN


sub.head()
new_test.head()



sub.tail()
new_test.tail()

########## MEDIAS NACHO ##########
lista_medias = [medias] * len(y_test)
array_medias = np.asarray(lista_medias).astype(np.float32) 
roc_auc_score(y_true = y_test, y_score = array_medias)
roc_auc_score(y_true = y_test, y_score = a)



exit()





















