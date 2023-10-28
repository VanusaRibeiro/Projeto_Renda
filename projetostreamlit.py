

import pandas as pd
import numpy as np
import streamlit as st
import datetime as dt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from xgboost import XGBClassifier
#import sklearn
import altair
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
# -------------------------------------- Cabeçalho -------------------------------------------------------#
sns.set(context='talk', style='ticks')

st.set_page_config(
    page_title="APR - Projeto Análise de Previsão de Renda",
    page_icon="📈",
)

st.title("🤑 APR - Análise Exploratória de Previsão de Renda")
st.header("")


with st.expander("ℹ️ - About this app", expanded=False):
	st.write(
        """     
	-   Esse app foi criado para realizar análise de renda de um grupo de pessoas com relação a Concessão de Cartão de Crédito.
    -   nosso objetivo é construir um modelo preditivo para identificar o risco de inadimplência (tipicamente definida pela   
        ocorrência de um atraso maior ou igual a 90 em um horizonte de 12 meses) através de variáveis que podem ser observadas 
        na data da avaliação do crédito (tipicamente quando o cliente solicita o cartão).
        desenvolver o melhor modelo preditivo de modo a auxiliar o mutuário a tomar suas próprias decisões referentes a crédito. 
        Nessa etapa também se avalia a situação da empresa/segmento/assunto de modo a se entender o tamanho do público, relevância, 
        problemas presentes e todos os detalhes do processo gerador do fenômeno em questão, e portanto dos dados.
	-   O código do app pode ser conferido em: 'https://github.com/VanusaRibeiro/Projeto/blob/main/renda%20previsao-Copy1.ipynb'
	-   Autora: Vanusa Ribeiro
	    """
	)
	st.markdown("")  

    
st.write('## Projeto baseado na Tabela abaixo')

renda = pd.read_csv('./Downloads/colab/projeto2023/projeto final/projeto 2/previsao_de_renda.csv')

@st.cache_resource
def load_data():
    # Carregue seu DataFrame aqui
    df = renda
    return df
 
df = load_data()
st.dataframe(df)
st.subheader('Com base na tabela mencionada foram feitos levantamentos estatísticos em que se obteve os seguintes insights:')
st.subheader('Em relação ao público foi identificado que:')
st.subheader('1- Públicos que possuem Nível Superior completo, Servidores Públicos, casados, possuem imóveis ou veículos e no máximo 02 filhos, tem menos riscos de inadimplência.')
st.subheader('2- Públicos empresários, assalariados,bolsistas, com nível médio,solteiros ou separados,possuem uma renda moderada não demonstrando um alto risco de inadimplência.')
st.subheader('3- Públicos pensionistas, viúvos, que moram com os pais e com maior quantidades de filhos possuem um alto risco de inadimpplência.')
st.subheader('Podemos verificar essas informações nos gráficos gerados ao longo do tempo:')
#plots
fig, ax = plt.subplots(8,1,figsize=(10,70))
renda[['posse_de_imovel','renda']].plot(kind='hist', ax=ax[0])

st.write('## Gráficos ao longo do tempo')

sns.lineplot(x='data_ref',y='renda', hue='posse_de_imovel',data=renda, ax=ax[1])
ax[1].tick_params(axis='x', rotation=45)

sns.lineplot(x='data_ref',y='renda', hue='posse_de_veiculo',data=renda, ax=ax[2])
ax[2].tick_params(axis='x', rotation=45)

sns.lineplot(x='data_ref',y='renda', hue='qtd_filhos',data=renda, ax=ax[3])
ax[3].tick_params(axis='x', rotation=45)

sns.lineplot(x='data_ref',y='renda', hue='tipo_renda',data=renda, ax=ax[4])
ax[4].tick_params(axis='x', rotation=45)

sns.lineplot(x='data_ref',y='renda', hue='educacao',data=renda, ax=ax[5])
ax[5].tick_params(axis='x', rotation=45)

sns.lineplot(x='data_ref',y='renda', hue='estado_civil',data=renda, ax=ax[6])
ax[6].tick_params(axis='x', rotation=45)

sns.lineplot(x='data_ref',y='renda', hue='tipo_residencia',data=renda, ax=ax[7])
ax[7].tick_params(axis='x', rotation=45)
sns.despine()
st.pyplot(plt)

st.write('## Gráficos bivariada')

fig, ax = plt.subplots(7,1,figsize=(10,50))
sns.barplot(x='posse_de_imovel',y='renda',data=renda, ax=ax[0])
sns.barplot(x='posse_de_veiculo',y='renda',data=renda, ax=ax[1])
sns.barplot(x='qtd_filhos',y='renda',data=renda, ax=ax[2])
sns.barplot(x='tipo_renda',y='renda',data=renda, ax=ax[3])
sns.barplot(x='educacao',y='renda',data=renda, ax=ax[4])
sns.barplot(x='estado_civil',y='renda',data=renda, ax=ax[5])
sns.barplot(x='tipo_residencia',y='renda',data=renda, ax=ax[6])
sns.despine()
st.pyplot(plt)