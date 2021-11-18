import re
import html
import nltk
import pandas as pd
nltk.download('stopwords')
from nltk import ngrams
from nltk.tokenize import RegexpTokenizer
from geopy.geocoders import ArcGIS
from geopy.extra.rate_limiter import RateLimiter
from tqdm import tqdm



#Primeiramente vou trar os dados que estão embutidos no HTML
#Função remove_html para extratir as informações do html
#função auxiliar apply_lambda para aplicar o lambda e retornar os dados 
#Manassés
def clean_text(string) -> str:
  regex_html_tag = re.compile('<.*?>')
  regex_b_lash_c = r'[^ \w\.]'
  html_unescape = html.unescape(string)
  remove_tags = re.sub(regex_html_tag, '', str(html_unescape))
  remove_b_n = re.sub(regex_b_lash_c, '', remove_tags)
  remove_d = remove_b_n.replace("-", "")

  return remove_d



def remove_null_empty_rows(data, column):
    data[column].replace('', np.nan, inplace=True)
    return data.dropna(subset=[column], inplace=True)


def most_commun_string(data, quant_words, language):
  tokenizer = RegexpTokenizer(r'\w+')
  #Restringindo apenas para os 1000 primeiras linhas
  data_tokenized = data.iloc[0:1000].apply(lambda value: tokenizer.tokenize(str(value)))
  #Tranformando de (lista de listas) para apenas uma lista e no processo transformando as palavras para minusculo
  vals = [word.lower() for lst in data_tokenized for word in lst]
  #retirando as "stopwords" em portugues 
  stopwords = nltk.corpus.stopwords.words(language)
  allWordExceptStop = [w for w in vals if w not in stopwords]
  #Aplicando o ngram e transformando em um pandas dataframe
  most_commun_phrases = pd.Series([q_word for n in quant_words for q_word in ngrams(allWordExceptStop, n)]).value_counts().reset_index(name="count")
  return most_commun_phrases

#Romario
def padronizacao_escrita(dataframe, nomeColuna):
  dataframe[nomeColuna] = dataframe[nomeColuna].str.normalize('NFKD').str.encode('ascii', errors='ignore').str.decode('utf-8')
  dataframe[nomeColuna]=dataframe[nomeColuna].str.upper()


def correcao_bairros(dataframe):
  ##retira acentos, deixa tudo em uppercase, retira das strings nomes como ap, apt quando foram preenchidas com apartamento
  aux_functions.padronizacao_escrita(dataframe, "bairro")

  ##retira os dados de apartamento que existe no código
  dataframe.bairro = dataframe.bairro.str.replace('\sAP\d', ' ', regex=True)
  dataframe.bairro = dataframe.bairro.str.replace('\sAP\s\d', ' ', regex=True)
  dataframe.bairro = dataframe.bairro.str.replace('\sAPT\s\d', ' ', regex=True)
  dataframe.bairro = dataframe.bairro.str.replace('\sAPT\d', ' ', regex=True)

  # gera uma copia do dataframe com uma nova coluna bairro_corrigido, preparando ela para ser aplicado o método de correção
  dataframeCopia = dataframe.dropna(axis=0, subset=["bairro"])
  dataframeCopia['bairro_corrigido'] = dataframe['bairro'] + ' ' + dataframe['uf'] + ' ' + 'Brasil'

  # Agrupa de forma que tenhamos apenas valores únicos antes de aplicar o método
  dataframeAgrupado = dataframeCopia.groupby(['bairro_corrigido', 'bairro']).size().reset_index(
    name='counts').sort_values(by=['counts'], ascending=False)

  # inicaliza o serviço ArcGIS a partir do geopy
  tqdm.pandas()
  geolocator = ArcGIS(user_agent="ampliar", timeout=10000)
  geocode = RateLimiter(geolocator.geocode, min_delay_seconds=2, max_retries=1, swallow_exceptions=True,
                        return_value_on_exception=None)

  # Aplica o método, colocando os valores corretos de bairro em bairro_corrigido
  dataframeAgrupado['bairro_corrigido'] = dataframeAgrupado['bairro_corrigido'].progress_apply(geolocator.geocode)
  dataframeAgrupado['bairro_corrigido'] =  dataframeAgrupado['bairro_corrigido'].astype('str').str.strip().astype('str').str.split(",").str[0]
  dataframe = pd.merge(dataframe, dataframeAgrupado[["bairro", "bairro_corrigido"]], on="bairro")

  # Retira novamento acentos, e deixa tudo em uppercase, para garantir que tudo siga o mesmo padrão
  aux_functions.padronizacao_escrita(dataframe, 'bairro_corrigido')

  return dataframe


# Gabriel

# Função que realiza o pré-processamento da coluna 'idade/dt. nasc' do banco de dados
def preProcessamentoDate(dataFrame, nomeColuna, ano):
  dataFrame[nomeColuna] = pd.to_datetime(dataFrame[nomeColuna], errors='coerce') # converte a coluna com dados no formato de data para datatime, o parametro erros='coerce' joga os dados que estão fora do range do datatime para NaN
  dataFrame['Ano'] = dataFrame[nomeColuna].dt.year.astype('Int64')
  dataFrame['Mes'] = dataFrame[nomeColuna].dt.month.astype('Int64')
  dataFrame['idade'] = ano - dataFrame['Ano']
  dataFrame['idade'] = dataFrame['idade'].astype('Int64')
  dataFrame = dataFrame[dataFrame.idade > 0]


#Função que preprocessa a coluna paciente_solicita_exame_site
def preProcssSolicitaExame(dataFrame, nomeColuna):
  dataFrame.loc[dataFrame[nomeColuna]> 1] = 1
  dataFrameAgrupadoPorSolicitaExame = dataFrame.groupby(nomeColuna, sort=True).size().reset_index(name='counts').sort_values(by=['counts'], ascending=False).reset_index(drop='index')
  QntPedidosExameSite = dataFrameAgrupadoPorSolicitaExame.iat[0,1]
  dataFrameAgrupadoPorSolicitaExame.loc[1] = [ 'NaN', (len(dataFrame[nomeColuna]) - QntPedidosExameSite)]
  return dataFrameAgrupadoPorSolicitaExame


#Função que remove tags html das colunas descritivas
def removeHtmlTags(dataFrame, nomeColuna):
  dataFrame[nomeColuna] = dataFrame[nomeColuna].astype('str')
  dataFrame[nomeColuna] = dataFrame[nomeColuna].str.replace(r'<[^<>]*>', '', regex=True).str.strip().str.replace(
  r'\n', '', regex=True).str.replace(r'\r', '', regex=True).str.replace(r'\r*', '', regex=True).str.replace('*', '', regex=True)
  dataFrame[nomeColuna] = dataFrame[nomeColuna].apply(lambda x: html.unescape(x))


#Agrupa o dataframe em relação a um atributo
def agrupaDataFrame(dataFrame, nomeColuna):
  return dataFrame.groupby(nomeColuna, sort=True).size().reset_index(name='counts').sort_values(by=['counts'], ascending=False).reset_index(drop='index')


# Vinicius

def cleaning_date_created(df):
  # Exclui linhas sem informação relevante 
  df.dropna(subset=['date_created'], inplace=True)
  df.drop(index=df.sort_values('date_created').head(18).index, inplace=True)
  df.drop(df.sort_values('date_created').tail(9).index, inplace=True)

  # Conversão de tipo str -> datetime
  df['date_created'] = pd.to_datetime(df['date_created']).dt.normalize()

  return df


def preplot_dados_ausentes(df):
  # Isola "ano" em uma coluna para facilitar o agrupamento dos dados
  ampliar_null = df.copy()
  ampliar_null['year_created'] = ampliar_null['date_created'].dt.year

  # Soma dos valores ausentes por coluna, agrupados por ano
  ampliar_null = ampliar_null.groupby('year_created').count().rsub(ampliar_null.groupby('year_created').size(), axis=0)
  ampliar_null.index.name = None
  ampliar_null['total'] = ampliar_null.sum(axis=1)

  # Soma dos registros, agrupados por ano
  registros_por_ano = {}

  for ano in range(2006,2022):
    registros_por_ano[ano] = df[df['date_created'].dt.year == ano].shape[0]

  # Concatenação dos DataFrames 
  total_nulos_df = ampliar_null[['total']]
  total_registros_df = pd.DataFrame.from_dict(data=registros_por_ano, 
                                              orient='index',
                                              columns=['total_registros'])
  total_registros_e_nulos = pd.concat([total_nulos_df, total_registros_df], axis=1)

  return total_registros_e_nulos


def cleaning_codigo_barra(df):
  # Exclui linhas com valores ausentes 
  df.dropna(subset=['codigo_barra'], inplace=True)
  
  # Exclui registros duplicados
  df = df.drop_duplicates(subset=['codigo_barra'])

  # Conversão de tipo float -> str
  df.loc[:, 'codigo_barra'] = df['codigo_barra'].astype(str)

  return df


def cleaning_data_recepcao(df):
  # Conversão de tipo str -> datetime
  df['data_recepcao'] = pd.to_datetime(df["data_recepcao"], errors="coerce").dt.normalize()

  # Preenche valores ausentes com valor de `date_created`
  df['data_recepcao'] = df['data_recepcao'].fillna(df['date_created'])

  return df


def cleaning_material_recebido(df):
  # Remover as tags html
  df['material_recebido'] = df['material_recebido'].str.replace(r'<[^<>]*>', '', regex=True)

  # Remover caracteres especiais
  df['material_recebido'] = df['material_recebido'].astype(str).apply(lambda x: html.unescape(x))

  # Troca quebras de linha, returns e EOF's por espaço em branco
  df['material_recebido'] = df['material_recebido'].str.replace(r'[\r|\n|\r\n|\xa0]+', ' ', regex=True)

  # Converte strings vazias para NaN
  df['material_recebido'].replace("", np.nan, inplace = True)
  df['material_recebido'].replace(" ", np.nan, inplace = True)
  df['material_recebido'].replace("nan", np.nan, inplace = True)

  # Preenche valores ausentes com a str "sem registro"
  df['material_recebido'] = df['material_recebido'].fillna("Sem registro")

  # Converte para caixa-baixa
  df['material_recebido'] = df['material_recebido'].str.lower()
  
  return df


def cleaning_total_pagar(df):
  # Concersão de tipo str -> float
  df['total_pagar'] = pd.to_numeric(df['total_pagar'])

  # Preenche valores ausentes com -1
  df['total_pagar'] = df['total_pagar'].fillna(-1)

  return df


def cleaning_exame_id(df):
  # Preenche valores ausentes com '0000000'
  df['exame_id'] = df['exame_id'].fillna('0000000')

  return df

##Yasmin
#Função que irá substituir todas as linhas que não possuem informação ou códigos aleatórios por "sem local definido"
def substitui(df, coluna):
  df[coluna].fillna('*Sem local definido*', inplace=True)
  df[coluna]= df[coluna].astype(str)
  df.loc[(df[coluna] == '143482') | (df[coluna] == '.') | (df[coluna] == 'F') | 
       (df[coluna] == '1270990') | (df[coluna] == ',') | (df[coluna] == 'C35014269 | C35') | 
       (df[coluna] == '1270990') | (df[coluna] == ',') | (df[coluna] == '0') | 
       (df[coluna] == '38612') | (df[coluna] == '65182')|(df[coluna] == '0.0') | 
       (df[coluna] == '38612.0') | (df[coluna] == '30955.0') | (df[coluna] == '30957.0') | 
       (df[coluna] == '30968.0') | (df[coluna] == '30963.0') | (df[coluna] == '30965.0') | 
       (df[coluna] == '30967.0') | (df[coluna] == '30965.0') | 
        (df[coluna] == '30957.0') | (df[coluna] == '30973.0'), coluna]='*Sem local definido*'
  return df

#Função que irá deletar as colunas sem muita relevância
def ApagaCol(df, coluna):
  df = df.drop(df[coluna], axis =1)
  return df


# Rafael

# A funcao abaixo altera os tipos iniciais dos dados de 'object' para 'datetime'

def altera_tipo_para_datetime(dataframe):
  for column in dataframe:
    dataframe[column] = pd.to_datetime(dataframe[column], errors = 'coerce')      # o parametro errors='coerce' diz que dados fora do range do tipo datetime recebem NaN
  
def printa_descricoes(dataframe):
    for column in dataframe:
        print('Descricao da coluna', column, ':')
        print(dataframe[column].describe())
        print()
        print(dataframe[column].describe(datetime_is_numeric=True))
        print()
        print()


# Pedro 

  #função que remove os ruidos de html.
def removeHtml(dataframe,coluna):
    dataframe[coluna] = dataframe[coluna].astype('str')
    dataframe[coluna] = dataframe[coluna].str.replace(r'<[^<>]*>', '', regex=True).str.strip().str.replace(
    r'\n', '', regex=True).str.replace(r'\r', '', regex=True).str.replace(r'\r*', '', regex=True).str.replace('*', '', regex=True)
    dataframe[coluna] = dataframe[coluna].apply(lambda x: html.unescape(x))

  #funções que limpam dados nulos e estranhos das colunas diagnostico_clinico e dados_clinicos.

def limpa_dados_clinicos(remove_nulos):
#colocando todos os dados estranhos em uma só variavel.
    dados_estranhos = remove_nulos.loc[(remove_nulos.dados_clinicos == '-')|(remove_nulos.dados_clinicos == '')|(remove_nulos.dados_clinicos == '- ')|(remove_nulos.dados_clinicos == 'None')|
  (remove_nulos.dados_clinicos == ' ')|(remove_nulos.dados_clinicos =='-.')|(remove_nulos.dados_clinicos == '-.')|(remove_nulos.dados_clinicos == '-3')|(remove_nulos.dados_clinicos == 'F-')
  |(remove_nulos.dados_clinicos == '-+')|(remove_nulos.dados_clinicos == '-C')|(remove_nulos.dados_clinicos == '-L')|(remove_nulos.dados_clinicos == '-ão')|(remove_nulos.dados_clinicos == 'h-')
  |(remove_nulos.dados_clinicos == ',-')|(remove_nulos.dados_clinicos == 'h')|(remove_nulos.dados_clinicos == 'm')|(remove_nulos.dados_clinicos == '-')|(remove_nulos.dados_clinicos == '-s')
  |(remove_nulos.dados_clinicos == '--')|(remove_nulos.dados_clinicos == '-\\')|(remove_nulos.dados_clinicos == '-H')|(remove_nulos.dados_clinicos == 'a,b-')|(remove_nulos.dados_clinicos == 'p-')
  |(remove_nulos.dados_clinicos == '\\-')|(remove_nulos.dados_clinicos == 'Pó')|(remove_nulos.dados_clinicos == 'Lo-')|(remove_nulos.dados_clinicos == '-E')|(remove_nulos.dados_clinicos == '-M')
  |(remove_nulos.dados_clinicos == '-l')|(remove_nulos.dados_clinicos == 'Ve')|(remove_nulos.dados_clinicos == '-o')|(remove_nulos.dados_clinicos == '-\\')|(remove_nulos.dados_clinicos == '-\xa0')
  |(remove_nulos.dados_clinicos == '\xa0')|(remove_nulos.dados_clinicos == ',')]
  #remoção das linhas do dataframe com os dados indexados na variavel dados_estranhos.
    remove_dado_estranho= remove_nulos.drop(dados_estranhos.index)
    limpeza_completa_dados = remove_dado_estranho.dropna(subset = ['dados_clinicos'])
    return limpeza_completa_dados

def limpa_diagnostico_clinico(remove_nulos):
#colocando todos os dados estranhos em uma só variavel.
  diagnosticos_estranhos = remove_nulos.loc[(remove_nulos.diagnostico_clinico == '-')|(remove_nulos.diagnostico_clinico == '')|(remove_nulos.diagnostico_clinico == '- ')
  |(remove_nulos.diagnostico_clinico == 'None')|(remove_nulos.diagnostico_clinico == '-.')|(remove_nulos.diagnostico_clinico == 'H-')|(remove_nulos.diagnostico_clinico == '- [')
  |(remove_nulos.diagnostico_clinico == '-9')|(remove_nulos.diagnostico_clinico == '-1')|(remove_nulos.diagnostico_clinico == '-,')|(remove_nulos.diagnostico_clinico == '-,3')
  |(remove_nulos.diagnostico_clinico == '-2')|(remove_nulos.diagnostico_clinico == '-=')|(remove_nulos.diagnostico_clinico == '-nm')|(remove_nulos.diagnostico_clinico == '-06')
  |(remove_nulos.diagnostico_clinico == ',-')|(remove_nulos.diagnostico_clinico == '--')|(remove_nulos.diagnostico_clinico == '-E:')|(remove_nulos.diagnostico_clinico == '-cistificado')
  |(remove_nulos.diagnostico_clinico == '-mo espia')|(remove_nulos.diagnostico_clinico == '-especificad')|(remove_nulos.diagnostico_clinico == '-O mesmo esp')
  |(remove_nulos.diagnostico_clinico == '-O mesmo especifica')|(remove_nulos.diagnostico_clinico == '.')|(remove_nulos.diagnostico_clinico == 'E-')|(remove_nulos.diagnostico_clinico == 'h--')
  |(remove_nulos.diagnostico_clinico == '-SPAC')|(remove_nulos.diagnostico_clinico == '- espe')|(remove_nulos.diagnostico_clinico == "-'")|(remove_nulos.diagnostico_clinico == '-02')
  |(remove_nulos.diagnostico_clinico == '- ')|(remove_nulos.diagnostico_clinico == '-+')|(remove_nulos.diagnostico_clinico == '-S')|(remove_nulos.diagnostico_clinico == '-6')
  |(remove_nulos.diagnostico_clinico == '-ecificado')|(remove_nulos.diagnostico_clinico == '-21')|(remove_nulos.diagnostico_clinico == '-18')|(remove_nulos.diagnostico_clinico == 'mAMA')
  |(remove_nulos.diagnostico_clinico == '-4')|(remove_nulos.diagnostico_clinico == 'u')|(remove_nulos.diagnostico_clinico == '- ')|(remove_nulos.diagnostico_clinico == '-\xa0')]
#remoção das linhas do dataframe com os dados indexados na variavel diagnosticos_estranhos.
  remove_diag_estranho = remove_nulos.drop(diagnosticos_estranhos.index)
#remoção das linhas nulas.
  limpeza_completa_diagnostico = remove_diag_estranho.dropna(subset = ['diagnostico_clinico'])

  return limpeza_completa_diagnostico

def ajusta_coluna_cobrar_de(dataframe):
#coloca 'desconhecido' nas linhas nulas.
  dataframe['cobrar_de'] = dataframe['cobrar_de'].fillna("Desconhecido")
#agrupamento dos itens presentes na coluna cobrar_de.
  agrupado = dataframe.groupby('cobrar_de', sort=True).size().reset_index(name='counts').sort_values(by=['counts'], ascending=False).reset_index(drop='index')
  return agrupado



# Walmir

def convert_to_int(dataframe, coluna):
  dataframe[coluna] = pd.to_numeric(dataframe[coluna], errors='coerce')
  dataframe[coluna] = dataframe[coluna].astype('Int64') 
  return dataframe

def qntd_linhas_faltantes(dataframe):
  return dataframe.isna().mean().reset_index(name="vazio").sort_values(by="vazio", ascending=False).reset_index(drop='index')

def clean_carater_atendimento(dataframe, coluna):
  dataframe[coluna] = dataframe[coluna].fillna("Sem descrição")
  return dataframe

def clean_missing_id(dataframe, listColumns):
  for col in listColumns:
    dataframe[col].fillna(-1, inplace=True)

  return dataframe

