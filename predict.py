import pandas as pd
import pickle
import statistics as st
import pyarrow.parquet as pq
from datetime import datetime

from sklearn.feature_extraction.text import CountVectorizer


def result(df):
  lista = []
  votacao = []
  for i in df.index:
    resultados = list(df.iloc[i,:].unique())
    if len(resultados) <= 2:
      lista.append(st.mode(list(df.iloc[i,:])))
      votacao.append(1)
    else:
      lista.append(-1)
      votacao.append(0)
  return lista, votacao


def importing():
  #Importando Base de dados Não Classificada
  df = pq.ParquetFile('input/n_classificado.parquet').read()
  return df.to_pandas()


def load_categories():
  #Importando Categorias
  return pd.read_json('categories/categorias.json')


def load_models():
  #Import Models
  with open("models/model1.pkl", "rb") as file1:#models/model1.pkl
      model1 = pickle.load(file1)

  with open("models/model2.pkl", "rb") as file2:#models/model2.pkl
      model2 = pickle.load(file2)

  with open("models/model3.pkl", "rb") as file3:#models/model3.pkl
      model3 = pickle.load(file3)

  return model1, model2, model3


def load_vectors():
  #Import Vectors
  with open("vectors/vector1.pkl", "rb") as vec1:#vectors/vector1.pkl
      vector1 = pickle.load(vec1)

  with open("vectors/vector2.pkl", "rb") as vec2:#vectors/vector2.pkl
      vector2 = pickle.load(vec2)

  with open("vectors/vector3.pkl", "rb") as vec3:#vectors/vector3.pkl
      vector3 = pickle.load(vec3)

  return vector1, vector2, vector3


def get_X(df):
  return df[df.columns[1]]


def m1(X, model1, vector1):
  #Predição modelo 1
  vector1 = vector1.transform(X)
  return model1.predict(vector1)
  

def m2(X, model2, vector2):
  #Predição modelo 2
  vector2 = vector2.transform(X)
  return model2.predict(vector2)


def m3(X, model3, vector3):
  #Predição modelo 3
  vector3 = vector3.transform(X)
  return model3.predict(vector3)


def selecting(df_pred, categorias):
  df_definido = df_pred[df_pred['votacao'] == 1]
  df_indefinido = df_pred[df_pred['votacao'] == 0]

  df_definido['ds_estrutura_mercadologica4'] = [i.split('|')[0] for i in df_definido['resultado']]
  df_definido['ds_estrutura_mercadologica5'] = [i.split('|')[1] for i in df_definido['resultado']]
  df_definido.drop(columns=['resultado','votacao'], axis=1, inplace=True)

  df_definido = df_definido.merge(categorias, how='left', left_on='ds_estrutura_mercadologica4', right_on='ds_estrutura_mercadologica4')

  df_cat = pd.DataFrame({
    'cd_ean': df_definido['cd_ean'],
    'descricao': df_definido['descricao'],
    'ds_estrutura_mercadologica1': df_definido['ds_estrutura_mercadologica1'],
    'ds_estrutura_mercadologica2': df_definido['ds_estrutura_mercadologica2'],
    'ds_estrutura_mercadologica3': df_definido['ds_estrutura_mercadologica3'],
    'ds_estrutura_mercadologica4' : df_definido['ds_estrutura_mercadologica4'],
    'ds_estrutura_mercadologica5': df_definido['ds_estrutura_mercadologica5']
  })

  df_n_cat = df_indefinido[['cd_ean','descricao']]

  return df_cat, df_n_cat


def voting(predicted1, predicted2, predicted3, X, df, categorias):
  df_pred = pd.DataFrame({'p1': [i for i in predicted1],
                         'p2': [i.split(">")[3] for i in predicted2],
                         'p3': [i for i in predicted3]})
  
  #Criação das colunas resultado e votação
  resultado, votacao = result(df_pred)

  df_pred['cd_ean'] = df['cd_ean']
  df_pred['descricao'] = X
  df_pred['resultado'] = resultado
  df_pred['votacao'] = votacao
  df_pred = df_pred[['cd_ean','descricao','resultado','votacao']]
  
  #Separando e tratando Dataframe entre votação definida e indefinida
  df_definido, df_indefinido = selecting(df_pred, categorias)

  return df_definido, df_indefinido


def save_results(df_definido, df_indefinido, a, b):
  df_definido.to_parquet("output/produtos_classificados.parquet")
  df_indefinido.to_parquet("output/produtos_indeterminados.parquet")
  relatorio(df_definido, df_indefinido, a, b)


def relatorio(df_definido, df_indefinido, a, b):
  
  try:
    df_json = pd.read_json("output/relatorio.json")
    relatorio = pd.DataFrame({
                      'qt_classificar': [df_definido.shape[0] + df_indefinido.shape[0]],
                      'qt_classificados': [df_definido.shape[0]],
                      'qt_indeterminados': [df_indefinido.shape[0]],
                      'dt_processamento': [datetime.now().strftime("%Y/%m/%d - %H:%M:%S")],
                      'ts_predict': [str(b - a).split(".")[0]]
    })
    df_json = pd.concat([df_json, relatorio], ignore_index=True)

  except:
    df_json = pd.DataFrame({
                      'qt_classificar': [df_definido.shape[0] + df_indefinido.shape[0]],
                      'qt_classificados': [df_definido.shape[0]],
                      'qt_indeterminados': [df_indefinido.shape[0]],
                      'dt_processamento': [datetime.now().strftime("%Y/%m/%d - %H:%M:%S")],
                      'ts_predict': [str(b - a).split(".")[0]]
    })
     
  df_json.to_json("output/relatorio.json")


def run():
  a = datetime.now()
  #Importando Base de dados Não Classificada
  df = importing()

  #Importando Categorias
  categorias = load_categories()

  #Import Models
  model1, model2, model3 = load_models()

  #Import Vectors
  vector1, vector2, vector3 = load_vectors()

  #Selecionando a coluna preditora
  X = get_X(df)

  #Prevendo resultado dos modelos
  predicted1 = m1(X, model1, vector1)
  predicted2 = m2(X, model2, vector2)
  predicted3 = m3(X, model3, vector3)

  #Votação de decisão
  df_definido, df_indefinido = voting(predicted1, predicted2, predicted3, X, df, categorias)

  #Salvando resultados
  b = datetime.now()
  save_results(df_definido, df_indefinido, a, b)



if __name__ == '__main__':
  run()