#%%
import os
import shutil

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
# from tensorflow.keras import layers
# from keras.models import load_model
from official.nlp import optimization  # to create AdamW optimizer
import tensorflow_datasets as tfds
tfds.disable_progress_bar()  

from official.modeling import tf_utils
from official import nlp
from official.nlp import bert
import re

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

tf.get_logger().setLevel('ERROR')

gpus = tf.config.experimental.list_physical_devices('GPU')

if gpus:
  # 텐서플로가 첫 번째 GPU만 사용하도록 제한
    tf.config.experimental.set_visible_devices(gpus[1], 'GPU')
    tf.config.experimental.set_memory_growth(gpus[1], True)
    
#%%
filepath = "./Data/EKG/BS2111_SM_Afib_train_2009.xlsx"

data = pd.read_excel(filepath)

# rsltcd_list = ['S035']

# for i in range(len(rsltcd_list)):
data['S035'] = (data.loc[:,'RSLT_CD_1':'RSLT_CD_7'] == 'S035').any(axis=1).astype(int)
# data['S024'] = (data.loc[:,'RSLT_CD_1':'RSLT_CD_7'] == 'S024').any(axis=1).astype(int)
# data['S003'] = (data.loc[:,'RSLT_CD_1':'RSLT_CD_7'] == 'S003').any(axis=1).astype(int)
data.columns

data = data.dropna(axis=0)

#%%
from sklearn.model_selection import train_test_split

train, val = train_test_split(data, test_size=0.1, random_state=1004, stratify=data['S035'])

bert_model_name = 'experts_wiki_books' 

map_name_to_handle = {
    'bert_en_uncased_L-12_H-768_A-12':
        'https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/3',
    'bert_en_cased_L-12_H-768_A-12':
        'https://tfhub.dev/tensorflow/bert_en_cased_L-12_H-768_A-12/3',
    'bert_multi_cased_L-12_H-768_A-12':
        'https://tfhub.dev/tensorflow/bert_multi_cased_L-12_H-768_A-12/3',
    'small_bert/bert_en_uncased_L-2_H-128_A-2':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-2_H-128_A-2/1',
    'small_bert/bert_en_uncased_L-2_H-256_A-4':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-2_H-256_A-4/1',
    'small_bert/bert_en_uncased_L-2_H-512_A-8':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-2_H-512_A-8/1',
    'small_bert/bert_en_uncased_L-2_H-768_A-12':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-2_H-768_A-12/1',
    'small_bert/bert_en_uncased_L-4_H-128_A-2':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-128_A-2/1',
    'small_bert/bert_en_uncased_L-4_H-256_A-4':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-256_A-4/1',
    'small_bert/bert_en_uncased_L-4_H-512_A-8':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-512_A-8/1',
    'small_bert/bert_en_uncased_L-4_H-768_A-12':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-768_A-12/1',
    'small_bert/bert_en_uncased_L-6_H-128_A-2':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-6_H-128_A-2/1',
    'small_bert/bert_en_uncased_L-6_H-256_A-4':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-6_H-256_A-4/1',
    'small_bert/bert_en_uncased_L-6_H-512_A-8':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-6_H-512_A-8/1',
    'small_bert/bert_en_uncased_L-6_H-768_A-12':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-6_H-768_A-12/1',
    'small_bert/bert_en_uncased_L-8_H-128_A-2':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-8_H-128_A-2/1',
    'small_bert/bert_en_uncased_L-8_H-256_A-4':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-8_H-256_A-4/1',
    'small_bert/bert_en_uncased_L-8_H-512_A-8':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-8_H-512_A-8/1',
    'small_bert/bert_en_uncased_L-8_H-768_A-12':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-8_H-768_A-12/1',
    'small_bert/bert_en_uncased_L-10_H-128_A-2':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-10_H-128_A-2/1',
    'small_bert/bert_en_uncased_L-10_H-256_A-4':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-10_H-256_A-4/1',
    'small_bert/bert_en_uncased_L-10_H-512_A-8':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-10_H-512_A-8/1',
    'small_bert/bert_en_uncased_L-10_H-768_A-12':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-10_H-768_A-12/1',
    'small_bert/bert_en_uncased_L-12_H-128_A-2':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-12_H-128_A-2/1',
    'small_bert/bert_en_uncased_L-12_H-256_A-4':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-12_H-256_A-4/1',
    'small_bert/bert_en_uncased_L-12_H-512_A-8':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-12_H-512_A-8/1',
    'small_bert/bert_en_uncased_L-12_H-768_A-12':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-12_H-768_A-12/1',
    'albert_en_base':
        'https://tfhub.dev/tensorflow/albert_en_base/2',
    'electra_small':
        'https://tfhub.dev/google/electra_small/2',
    'electra_base':
        'https://tfhub.dev/google/electra_base/2',
    'experts_pubmed':
        'https://tfhub.dev/google/experts/bert/pubmed/2',
    'experts_wiki_books':
        'https://tfhub.dev/google/experts/bert/wiki_books/2',
    'talking-heads_base':
        'https://tfhub.dev/tensorflow/talkheads_ggelu_bert_en_base/1',
}

map_model_to_preprocess = {
    'bert_en_uncased_L-12_H-768_A-12':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'bert_en_cased_L-12_H-768_A-12':
        'https://tfhub.dev/tensorflow/bert_en_cased_preprocess/3',
    'small_bert/bert_en_uncased_L-2_H-128_A-2':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-2_H-256_A-4':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-2_H-512_A-8':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-2_H-768_A-12':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-4_H-128_A-2':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-4_H-256_A-4':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-4_H-512_A-8':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-4_H-768_A-12':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-6_H-128_A-2':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-6_H-256_A-4':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-6_H-512_A-8':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-6_H-768_A-12':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-8_H-128_A-2':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-8_H-256_A-4':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-8_H-512_A-8':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-8_H-768_A-12':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-10_H-128_A-2':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-10_H-256_A-4':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-10_H-512_A-8':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-10_H-768_A-12':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-12_H-128_A-2':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-12_H-256_A-4':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-12_H-512_A-8':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-12_H-768_A-12':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'bert_multi_cased_L-12_H-768_A-12':
        'https://tfhub.dev/tensorflow/bert_multi_cased_preprocess/3',
    'albert_en_base':
        'https://tfhub.dev/tensorflow/albert_en_preprocess/3',
    'electra_small':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'electra_base':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'experts_pubmed':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'experts_wiki_books':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'talking-heads_base':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
}

tfhub_handle_encoder = map_name_to_handle[bert_model_name]
tfhub_handle_preprocess = map_model_to_preprocess[bert_model_name]

print(f'BERT model selected           : {tfhub_handle_encoder}')
print(f'Preprocess model auto-selected: {tfhub_handle_preprocess}')

bert_preprocess_model = hub.KerasLayer(tfhub_handle_preprocess)

#%%
def build_classifier_model():
    text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
    preprocessing_layer = hub.KerasLayer(tfhub_handle_preprocess, name='preprocessing')
    encoder_inputs = preprocessing_layer(text_input)
    encoder = hub.KerasLayer(tfhub_handle_encoder, trainable=True, name='BERT_encoder')
    outputs = encoder(encoder_inputs)
    net = outputs['pooled_output']
    net = tf.keras.layers.Dropout(0.1)(net)
    net = tf.keras.layers.Dense(1, activation='sigmoid', name='S035_class')(net)
    
#     for rslt_class in rsltcd_list:
#         globals()[rslt_class] = tf.keras.layers.Dense(1, activation='sigmoid', name='%s_class' % rslt_class)(net)

# #     net = tf.keras.layers.Dense(1, activation='sigmoid', name='classifier')(net)
    return tf.keras.Model(text_input, net)

#%%
classifier_model = build_classifier_model()
# bert_raw_result = classifier_model(tf.constant(test_text))
# print(tf.sigmoid(bert_raw_result))

# metrics = [tf.metrics.BinaryAccuracy(), tf.metrics.AUC(), tf.metrics.Recall()]
batch_size = 32
epochs = 3
# steps_per_epoch = len(train) // batch_size
steps_per_epoch = len(data) // batch_size
num_train_steps = steps_per_epoch * epochs
num_warmup_steps = int(0.1*num_train_steps)

init_lr = 3e-5
optimizer = optimization.create_optimizer(init_lr=init_lr,
                                          num_train_steps=num_train_steps,
                                          num_warmup_steps=num_warmup_steps,
                                          optimizer_type='adamw')

classifier_model.compile(optimizer=optimizer,
                         loss='binary_crossentropy',
                         metrics=['accuracy'])

classifier_model.summary()

# 데이터 학습실행
classifier_model.fit(train['검사결과내용7'], train['S035'],
                               batch_size=batch_size,
                               epochs=epochs)

#%%
classifier_model.evaluate(val['검사결과내용7'],  val['S035'])

# print(f'Loss: {loss}')
# print(f'Accuracy: {accuracy}')

#%%
# 학습시킨 모델 reload 후 inference

#%%
!mkdir -p saved_model

classifier_model.save('./Modelsave/EKG_Aflutter', include_optimizer=False)

#%%
ekg_predict = tf.keras.models.load_model('./Modelsave/EKG_Aflutter')

ekg_predict.summary()

#%%
for i in range(1999,2021):
    filepath = "./Data/EKG_multi/BS2111_{}.xlsx".format(i)

    df = pd.read_excel(filepath)
    
    df.rename(columns={"건강검진결과코드12":"RSLT_CD"}, inplace=True) 
    df['CON'] = df['환자번호1'] + df['SM_DATE']
    
    df['idx'] = df.groupby('CON').cumcount() + 1
    
    df = df.pivot_table(
                            index=['환자번호1','SM_DATE','CON','검사결과내용7'], columns='idx'
                           ,values=['RSLT_CD'], aggfunc='first'
                           )
    df = df.sort_index(axis=1, level=1)
    df.columns = [f'{x}_{y}' for x, y in df.columns]
    
    df = df.reset_index()
# 반복문 실행하니 해당되는 결과가 1개뿐이서 실행시 에러가 남. 전체 반복문에서 처리로 변경
#     for j in range(len(rsltcd_list)):
    df['S035'] = (df.loc[:,'RSLT_CD_1':df.columns[-1]] == 'S035').any(axis=1).astype(int)

    # reload한 모델로 prediction
    prediction = classifier_model.predict(df['검사결과내용7'])
    
#     for k in range(len(rsltcd_list)):
    rslt_pre = prediction.copy()
    rslt_pre = np.where(rslt_pre < 0.5 ,0, 1)
    df['S035_pre'] = rslt_pre
    df['S035_proba'] = prediction.copy()

#     # multi result prediction 결과 저장
#     with pd.ExcelWriter('./Data/EKG_multi/BS2111_multi_prediction_{}.xlsx'.format(i), mode='w', engine='openpyxl') as writer:
#             df.to_excel(writer, index=False)
            
    # A.flutter prediction 결과 저장
    globals()['ekg_flutter_{}'.format(i)] = df.loc[(df['S035'] == 1) | (df['S035_pre'] == 1)]
    with pd.ExcelWriter('./Data/EKG_flutter/{}_BS2111_flutter_prediction.xlsx'.format(i), mode='w', engine='openpyxl') as writer:
            globals()['ekg_flutter_{}'.format(i)].to_excel(writer, index=False)

