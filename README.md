PROJETO PREDIÇÃO DE CATEGORIAS DE PRODUTOS

Projeto de formação de Ciência de Dados da Escola DNC com empresa parceira

1. OBJETIVOS:
   Através de uma base de dados previamente classificada, construir um modelo de classificação de produtos para futuras bases de dados não classificadas.

2. ORGANIZAÇÃO DAS PASTAS E ARQUIVOS:
    .env  : Arquivos do Ambiente virtual

    categories  : Pasta de armazenamento do arquivo "categorias.json", onde é salvo uma planilha com todas as categorias utilizadas no treinamento do modelo. A cada vez que o modelo for treinado, o arquivo é atualizado de acordo com a nova base classificada de treinamento.

    input  : A pasta "input" é onde deve estar localizado o arquivo a ser classificado. O arquivo deve sempre possuir o nome "n_classificado.parquet", onde a primeira coluna constando o código de barras, "cd_ean", e a segunda coluna a descrição do produto a ser classificado.

    models  : Onde são guardados os modelos treinados. Quando o modelo for treinado novamente, estes serão sobrescritos por um novo modelo.

    output  : Onde são armazenados as planilhas com os arquivos classificados pelo modelo ("produtos_classificados.parquet"), os produtos onde o modelo não entrou em consenso na classificação do produto ("produtos_indeterminados.parquet"), e o arquivo "relatorio.json", onde constam a quantidade de prdutos de entrada, quantos foram preditos, quantos não puderam ser determinados, a data da predição e o tempo de predição.

    train_files  : Onde deverar constar os arquivo classificado toda vez que for necessário retreinar o modelo com uma nova base de dados classificada. Sempre deverá possuir o nome "classificado.parquet"

    vectors  : Onde estaram salvos as coleção de documentos de texto da matriz de contagens de token, geradas no treinamento do modelo e necessária para predição dos modelos. São sobrescritas cada vez que o modelo for retreinado.

    train.py  : Utilizado para treino e retreino do modelo.

    predict.py  : Utilizado para gerar as predições do modelo.

    