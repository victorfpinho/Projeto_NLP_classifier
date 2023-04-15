import pyarrow.parquet as pq
import pandas as pd
from random import sample
#Train_test_split
from sklearn.model_selection import train_test_split

#NLP
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import SGDClassifier

#Salvar Modelos
import pickle


#Função de Tratamento de valores duplicados e nulos
def tratar(df):
  '''
  df: Dataset a ser tratado

  Recebe um dataset e retorna-o sem valores nulos e repetidos
  '''


  df.drop_duplicates(subset=df.columns[1], keep='first', inplace=True)
  df = df[~df[df.columns[1]].isna()]
  df.reset_index(inplace=True)
  return df


#Função de Gerar índices aleatórios
def indexes(df):
  '''
  Recebe um df e separa seus índices em 2 grupos aleatoriamente
  '''
  return sample(range(0, df.shape[0]-1), int(df.shape[0]/2))


def model3(df):
  '''
  Retorna o modelo 3 treinado
  '''
  df_a, df_b = sep_df(df)
  modela, vectorizer = model_a(df_a)
  model_final, vectorizerc = model_b(df_b, modela, vectorizer)
  return (model_final, vectorizerc) 


def sep_df(df):
  '''
  Separa o dataset de entrada em 2 aleatoriamente
  '''
  #Gerando índices aleatórios
  idx = sample(range(0, df.shape[0]-1), int(df.shape[0]/2))

  #Separando o dataset em 2
  df_a = df[df.index.isin(idx)]
  df_b = df[~df.index.isin(idx)]

  return (df_a, df_b)


def model_a(df_a):
  '''
  Treina a primeira parte do dataset
  '''
  #Separação Train Test Split Cascata
  xa = df_a['ds_produto_mob2']
  ya = df_a['ds_estrutura_mercadologica1']

  x_traina, x_testa, y_traina, y_testa = train_test_split(xa, ya, test_size=0.2, random_state=42)

  #Converter uma coleção de documentos de texto em uma matriz de contagens de token
  vectorizer = CountVectorizer(ngram_range=(1,2))
  train_x_vectorsa = vectorizer.fit_transform(x_traina)


  #Treinamento do modelo com Support Vector Machine
  model_casc1 = SGDClassifier()
  return (model_casc1.fit(train_x_vectorsa, y_traina), vectorizer)


def model_b(df_b, model_casc1, vectorizer):
  '''
  Faz a predição da segunda parte do dataset, utilizando o treinamento da primeira parte do dataset, gerando um 
  terceiro dataset para predição final
  '''
  #predição do df_b
  xb = df_b['ds_produto_mob2']
  yb = df_b['ds_estrutura_mercadologica1']

  #Predição
  x_testb = vectorizer.transform(xb)
  predictedb = model_casc1.predict(x_testb)

  #Dataset com a descrição e a predição da mercadológica 1
  df_c = pd.DataFrame({'Produto': xb,
                       'ds_1_pred': predictedb})
  
  model_casc_final, vectorizerc = model_c(df_c, df_b, vectorizer)

  return (model_casc_final, vectorizerc)


def model_c(df_c, df_b, vectorizer):
  '''
  Retorna a predição final do modelo 3 (Abordagem em cascata)
  '''
  #Predição final
  yc = df_b['ds_estrutura_mercadologica4'] + '|' + df_b['ds_estrutura_mercadologica5']
  xc = df_c['Produto'] +' '+ df_c['ds_1_pred']

  x_trainc, x_testc, y_trainc, y_testc = train_test_split(xc, yc, test_size=0.2, random_state=42)

  #Converter uma coleção de documentos de texto em uma matriz de contagens de token
  train_x_vectorsc = vectorizer.fit_transform(x_trainc)

  #Treinamento do modelo com Support Vector Machine
  model_casc_final = SGDClassifier()
  return (model_casc_final.fit(train_x_vectorsc, y_trainc), vectorizer)


def model1(df):
  #Separando as bases de treino e teste do modelo 1 (Predição mercadológica 4 e 5)
  X = df['ds_produto_mob2']
  y_1 = df['ds_estrutura_mercadologica4'] + '|' + df['ds_estrutura_mercadologica5']
  x_train, x_test, y_train, y_test = train_test_split(X, y_1, test_size=0.2, random_state=42)

  vectorizer = CountVectorizer(ngram_range=(1,2))

  model_pipeline1 = SGDClassifier()
  x_train_vec1 = vectorizer.fit_transform(x_train)
  return (model_pipeline1.fit(x_train_vec1, y_train), vectorizer)


def model2(df):
  #Separando as bases de treino e teste do modelo 2 (Rígida)
  X = df['ds_produto_mob2']
  y_2 = pd.DataFrame(
      df['ds_estrutura_mercadologica1'] + '>' +
      df['ds_estrutura_mercadologica2'] + '>' +
      df['ds_estrutura_mercadologica3'] + '>' +
      df['ds_estrutura_mercadologica4'] + '|' +
      df['ds_estrutura_mercadologica5']
  )
  x_train2, x_test2, y_train2, y_test2 = train_test_split(X, y_2, test_size=0.2, random_state=42)

  #Converter uma coleção de documentos de texto em uma matriz de contagens de token
  vectorizer = CountVectorizer(ngram_range=(1,2))

  #Treinando o modelo
  model_pipeline2 = SGDClassifier()
  x_train_vec2 = vectorizer.fit_transform(x_train2)
  return (model_pipeline2.fit(x_train_vec2, y_train2), vectorizer)


def importing():
  #Importando Base de dados Classificadas
  df = pq.ParquetFile('train_files/classificado.parquet').read()
  new_names = ["cd_ean", "ds_produto_mob2" ,"ds_estrutura_mercadologica1","ds_estrutura_mercadologica2","ds_estrutura_mercadologica3","ds_estrutura_mercadologica4","ds_estrutura_mercadologica5"]
  try:
    df = df.rename_columns(new_names)
  except:
    pass

  return df.to_pandas()


def categorized(df):
  df = df[[
    'ds_estrutura_mercadologica1', 'ds_estrutura_mercadologica2',
      'ds_estrutura_mercadologica3','ds_estrutura_mercadologica4'
    ]]

  df = df.drop_duplicates(subset=['ds_estrutura_mercadologica4'])
  df.to_json("categories/categorias.json")


def serialize_object(model_pipeline1, model_pipeline2, model_pipeline3):
  #Serializar objeto
  #Modelo1
  with open('models/model1.pkl', 'wb') as file_model1:
    pickle.dump(model_pipeline1, file_model1)

  #Modelo2
  with open('models/model2.pkl', 'wb') as file_model2:
    pickle.dump(model_pipeline2, file_model2)

  #Modelo3
  with open('models/model3.pkl', 'wb') as file_model3:
    pickle.dump(model_pipeline3, file_model3)


def vectorize_object(vectorizer1, vectorizer2, vectorizer3):
  #Modelo1
  with open('vectors/vector1.pkl', 'wb') as vector1:
    pickle.dump(vectorizer1, vector1)

  #Modelo2
  with open('vectors/vector2.pkl', 'wb') as vector2:
    pickle.dump(vectorizer2, vector2)

  #Modelo3
  with open('vectors/vector3.pkl', 'wb') as vector3:
    pickle.dump(vectorizer3, vector3)


def run():
  #Importano base de dados
  df = importing()

  #Limpando e Tratando o Dataset
  df = tratar(df)

  #Categorizar
  categorized(df)

  #Model1
  model_pipeline1, vectorizer1 = model1(df)

  #Model2
  model_pipeline2, vectorizer2 = model2(df)

  #Model3
  model_pipeline3, vectorizer3 = model3(df)


  vectorize_object(vectorizer1, vectorizer2, vectorizer3)
  serialize_object(model_pipeline1, model_pipeline2, model_pipeline3)



if __name__ == '__main__':
  run()