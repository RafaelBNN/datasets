import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
import pandas as pd


def plot_most_commun_phrases(data, x, y):
  fig = px.bar(data.iloc[0:20, :], x=x, y=y, text=y)
  fig.update_traces(texttemplate='%{text:.2s}', textposition='outside')
  fig.show()

#função Victor Modesto
#devido a natureza dos valores das colunas(atipia, representacao, malignidade e sigla) não foi necessário um pré processamento, apenas o groupby do pandas já realizou o trabalho assim
#decidi fazer uma função generalista para a exposição desses dados.
def informacoes_coluna(df,coluna):
  print("Linhas Faltantes: ",dataframe[coluna].isnull().sum())
  print("Porcentagem de falta: ", dataframe[coluna].isnull().sum()/len(dataframe)*100,"%")
  agrupamentocoluna = dataframe.groupby([coluna]).size().reset_index(name='counts').sort_values(by=['counts'], ascending=False)
  #tabela para coluna
  print("Dados da Coluna ",coluna)
  fig = go.Figure(data=[go.Table(
    header=dict(values=list(agrupamentocoluna.columns),
                fill_color='paleturquoise',
                align='left'),
    cells=dict(values=[agrupamentocoluna[coluna], agrupamentocoluna.counts],
               fill_color='lavender',
               align='left'))
      ])
  fig.update_layout(
    title_text="Valores Representados por "+coluna
  )
  fig.show()
  #gráfico de pizza dos valores da tabela
  fig = px.pie(agrupamentocoluna, values= 'counts', names= coluna, title="Comparativos:")
  fig.show()
  #gráfico de barras
  fig = px.bar(agrupamentocoluna, x=coluna, y='counts', title="Quantitativos:")
  fig.show()

##Romario

#Função recebe dataframe, Lista com nome das coluna que desejasse plotar
def plotagem_basica_tabela(dataframe, nomesColuna):
  dataframeAgrupado = dataframe.groupby(nomesColuna).size().reset_index(name='counts').sort_values(by=['counts'],
                                                                                                   ascending=False)
  nomesColuna.append('counts')

  valoresPlotagem = []
  for i in range(len(nomesColuna)):
    valoresPlotagem.append(dataframeAgrupado[nomesColuna[i]])

  fig = go.Figure(data=[go.Table(
    header=dict(values=list(dataframeAgrupado.columns),
                fill_color='paleturquoise',
                align='left'),
    cells=dict(values=valoresPlotagem,
               fill_color='lavender',
               align='left'))
  ])


#Função recebe dataframe, Lista com nome das coluna que desejasse plotar, e a quantidade de barras que deseja-se no gráfico. Recomenda-se não colocar uma quantidade muito grande pois pode gerar problemas na plotagem
def plotagem_basica_graficodebarras(dataframe, nomeColuna, qtdBarras=None):
  # definindo titulo do gráfico
  titulo = "os " + str(qtdBarras) + " maiores valores de " + nomeColuna

  # Agrupando dados em relação a coluna de interesse que deseja-se plotar
  dataframeAgrupado = dataframe.groupby(nomeColuna).size().reset_index(name='counts').sort_values(by=['counts'],
                                                                                                  ascending=False)

  # verificando se não está tentando plotar mais colunas que o possível e gerando gráfico apenas para quantidade possível de barras.
  if qtdBarras!=None:
    if qtdBarras < dataframeAgrupado.shape[0]:
      dataframePlotagem = dataframeAgrupado.iloc[range(int(qtdBarras))]
    else:
      dataframePlotagem = dataframeAgrupado

    dataframePlotagem[nomeColuna] = dataframePlotagem[nomeColuna].astype('str')
  else:
    dataframePlotagem[nomeColuna] = dataframeAgrupado[nomeColuna].astype('str')

  # Criando Figura do plotly.express
  fig = px.bar(dataframePlotagem, y='counts', x=nomeColuna, text='counts', title=titulo)
  fig.update_traces(texttemplate='%{text:.2s}', textposition='outside')
  fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')
  fig.show()

# Gabriel
 
#Função que seleciona o tipo de gráfico

def tipoDeGraficos(dataFrame, nomeColuna, tipoDoGrafico, title=None):

  if tipoDoGrafico== 'Pizza':
    fig = px.pie(
                  dataFrame, values='counts', names=nomeColuna, title=title
              )
    fig.show()

  elif tipoDoGrafico== 'Barras':
    fig = px.bar(dataFrame, y='counts', x=nomeColuna, text='Counts', title=title)
    fig.update_traces(texttemplate='%{text:.2s}', textposition='outside')
    fig.update_layout(
                        yaxis_title='Quantidade' 
                      )
    fig.show()

#Função que imprime graficos e tabelas das colunas que tinham tags html para colunas com muitos valores diferentes

def imprimeTabelaColasDescritivas(dataFrame, nomeColuna):
  #Agrupa os dados do dataFrame em relação a nomeColuna
  dataFrameAgrupado = dataFrame.groupby(nomeColuna).size().reset_index(name='counts').sort_values(by=['counts'], ascending=False).reset_index(drop='index')
  
  #caso a quantidade de linhas do dataframe agrupado seja maior que 500 vamos imprimir somente os 16 primeiros que mais ocorreram
  if len(dataFrameAgrupado)>1000:
    dataFrameAgrupado = dataFrameAgrupado.iloc[[0,1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]]

  #Imprime a tabela 
  fig = go.Figure(data=[go.Table(
      columnwidth=[3, 1],
      header=dict(values=list(dataFrameAgrupado.columns),
                  fill_color='paleturquoise',
                  align='left'),
      cells=dict(values=[dataFrameAgrupado[nomeColuna], dataFrameAgrupado.counts],
                fill_color='lavender',
                align='left'))
  ])
  fig.show()


# Vinicius

def lineplot_dados_ausentes(df):
  '''Série temporal com evolução dos registros e valores ausentes.'''
  ax = plt.axes()
  ax.plot(df.index, df['total'], 
          lw=3, label='Valores ausentes')
  ax.plot(df.index, df['total_registros'],
          lw=3, label='Registros')
  ax.set(xlim=(2005, 2022), ylim=(0, 70000),
        xlabel='Ano', ylabel='Total',
        title='Valores ausentes vs nº de registros')
  ax.legend(frameon=False)
  plt.rcParams["figure.figsize"] = (12,8)

  plt.show()


def valores_ausentes_global(df):
  '''Visualização da relação entre valores ausentes nas variáveis.'''
  msno.matrix(df, figsize=(12,8))


def boxplot_receita(df):
  """Visualiza distribuição dos valores da variável total_pagar."""
  ampliar_receita = df[df['total_pagar'] > 0].copy()
  
  # Isola "ano" em uma coluna para facilitar o agrupamento dos dados
  ampliar_receita['year'] = ampliar_receita['date_created'].dt.year

  # Plot
  ax = plt.axes()
  ax.boxplot(ampliar_receita[ampliar_receita['year'] > 2018]['total_pagar'])
  ax.set(ylabel='Total pago (R$)', title='Distribuição dos valores recebidos (2019-2021)')
  plt.rcParams["figure.figsize"] = (12,8)
  
  plt.show()

##Yasmin
def GerarTabela(dataframe, nomesColuna):
  dataframeAgrupado = dataframe.groupby(nomesColuna).size().reset_index(name='counts').sort_values(by=['counts'],
                                                                                                   ascending=False)


  fig = go.Figure(data=[go.Table(
    header=dict(values=list(dataframeAgrupado.columns),
                fill_color='paleturquoise',
                align='left'),
    cells=dict(values= [dataframeAgrupado[nomesColuna], dataframeAgrupado.counts],
               fill_color='lavender',
               align='left'))
  ])
  fig.show()  

# Pedro  
#função que cria uma tabela com o dataframe e a coluna especificada.
def plot_tabela(dataframe,coluna):

  plot_tabela = dataframe[['id_paciente','sexo','coluna']]
  fig = go.Figure(data=[go.Table(
    header=dict(values=list(plot_tabela.columns),
                fill_color='paleturquoise',
                align='left'),
    cells=dict(values=[plot_tabela.id_paciente,plot_tabela.sexo, plot_tabela.coluna],
               fill_color='lavender',
               align='left'))
])
  fig.show()
# função que cria um gráfico de pizza com os valores especificados.
def plot_grafico(contador,coluna):
  fig = px.pie(contador, values='counts', names= coluna)

  fig.show()



# Walmir

def create_table(dataframe):
  fig = go.Figure(data=[go.Table(
    header=dict(values=list(dataframe.columns),
                fill_color='paleturquoise',
                align='left'),
    cells=dict(values=[dataframe.auxiliar_macroscopia_id, dataframe.medico_requisitante_id, dataframe.responsavel_macroscopia_id, dataframe.usuario_conclusao_id, dataframe.usuario_recepcao_id, dataframe.carater_atendimento],
               fill_color='lavender',
               align='left'))
])
  fig.show()