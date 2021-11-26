import streamlit as st
import numpy as np
import pandas as pd
import pickle
import shap

import plotly.express as px
import plotly.io as pio


# sidebar : pouvoir passer du dashboard au predicteur ml
sb = st.sidebar
rad_who = sb.radio('Application ML', ['Dashboard', 'Predictor'])

DATA_DIR_PATH = 'data/'

# partie dashboard
if rad_who == 'Dashboard':

    st.title('Dashboard Prêt à Dépenser')

    st.text("")
    st.text("")
    st.text("")
    
    pio.templates.default = 'seaborn'
    
    @st.cache
    def load_data(nrows):
    	data = pd.read_csv(DATA_DIR_PATH + 'Xpred_test_sub.csv', nrows=nrows, index_col = 0)
    	return data

    # charge 500 clients
    data = load_data(500)

    @st.cache
    def load_shap():
    	shap = pd.read_csv(DATA_DIR_PATH + 'shap_values_sub.csv', index_col = 0)
    	return shap
    
    # valeurs de shap associées au clients
    shap_values = load_shap()
    shap_values = shap_values.loc[data.index]
    
   
    # sélection d'un client particulier
    idclient = st.selectbox(
    		'Quel est l\'identifiant client ?',
    		data.index)
    
    st.write('Vous avez selectionné le client :', idclient)

    clientdata = data.loc[[idclient]]
    
    st.text("")
    st.text("")
    

    st.header('Probabilité de défaut de paiement')
    
    # affiche le score et la note client 
    col1, col2 = st.columns(2)
    col1.metric("Score (%)", round(clientdata['score'].values[0] * 100, 1) )
    col2.metric("Note", clientdata['note'].values[0])
    
    st.text("")
    st.write("Les 25% de clients aux scores les plus faibles ont une note A. Les 25% suivants ont une note de B, \
    et ainsi de suite jusqu'aux clients de note D.")
    st.text("")
    st.text("")
    

    ##################################################
    #                SHAPLEY VALUES
    ##################################################
    
    st.header('Interprétation Locale')
    st.write('Liste des variables expliquant le score du client ' , idclient, '. Leur importance relative pour ce client est exprimée en pourcentage.')
    
    
    # valeurs shap pour un client donné
    clientshap = shap_values.loc[idclient]
    
    # récupère le signe des valeurs shap
    clientshap_sgn = np.sign(clientshap).astype('int')
    clientshap_sgn = clientshap_sgn.replace([-1, 1], ['negative', 'positive'] )
    
    # bar plot des valeurs de shap normalisées
    fig = px.bar(x = abs(clientshap) / abs(clientshap).sum() * 100, 
                 y = clientshap.index, orientation='h',
                 color= clientshap_sgn
                )
    fig.update_layout(height=680, legend_title = 'Contribution',
                      yaxis_title_text = '',
                      xaxis_title_text = 'Shapley value (%)')
    fig.update_yaxes(categoryorder = 'total ascending')
    
    st.plotly_chart(fig)

    
    ##################################################
    #                VARIABLE DISTRIBUTION
    ##################################################
    
    # sélection d'une variable
    inputvar = st.selectbox(
    		'Variable du modèle à afficher',
    		data.columns)
    
    st.write('Vous avez selectionné:', inputvar)
    
    # valeur de la variable pour le client sélectionné
    xclient = clientdata[inputvar].values[0]
    
    # histogramme de la variable
    fig2 = px.histogram(data, x = inputvar, marginal='box')
    if inputvar != 'note':
        fig2.add_vline(x=xclient, line_width=3, line_dash="dash", line_color="red",
        )
    
    st.plotly_chart(fig2)
    
    # encodage
    if inputvar == 'CODE_GENDER':
        st.write('0 : Male, 1 : Female.')
    
    st.text("")
    st.text("")

    
    ##################################################
    #                FEATURE IMPORTANCE
    ##################################################
    
    st.header('Interprétation Globale')
    st.write('Importance des variables du modèles pour identifier les clients aux défauts de paiement.')
    
    # sélection des variables d'entrée
    df_v = data.drop(['score', 'note'], axis = 1)
    
    feature_list = df_v.columns
    
    # détermine le signe de la corrélation shap value et feature value 
    corr_list = list()
    # boucle sur les variables d'entrée
    for feat in feature_list:
        # retire les clients dont variable n'est pas renseignée (NaN)
        mask_notna = df_v[feat].notna()
        X_tmp = df_v[feat][mask_notna]

        shapi_tmp = shap_values[feat]
        idx_tmp = mask_notna[mask_notna].index
        shapi_tmp = shapi_tmp.loc[idx_tmp]

        # calcul du coefficient de corrélation
        b = np.corrcoef(shapi_tmp,X_tmp)[1][0]
        corr_list.append(b)

    corr_df = pd.concat([pd.Series(feature_list),pd.Series(corr_list)],axis=1).fillna(0)

    # Make a data frame. Column 1 is the feature, and Column 2 is the correlation coefficient
    corr_df.columns  = ['Variable','Corr']
    corr_df['Contribution'] = np.where(corr_df['Corr']>0,'positive','negative')
    
    shap_abs = np.abs(shap_values)

    # l'importance d'une variable définie comme la moyenne des |valeurs de Shapley|
    k=pd.DataFrame(shap_abs.mean()).reset_index()
    k.columns = ['Variable','SHAP_abs']
    k2 = k.merge(corr_df,left_on = 'Variable',right_on='Variable',how='inner')
    
    k2 = k2[k2["SHAP_abs"]!=0].tail(40)
    
    
    fig3 = px.bar(k2, x = 'SHAP_abs', y = 'Variable', 
                 color = 'Contribution', orientation='h')
    fig3.update_layout(height = 680,
                     yaxis_title_text = '',
                     xaxis_title_text = '|shap values| mean')
    fig3.update_yaxes(categoryorder = 'total ascending')
    
    st.plotly_chart(fig3)


# partie predicteur ml
if rad_who == 'Predictor':

    MODEL_DIR_PATH = 'model/'

    st.title('Prédicteur Prêt à Dépenser')

    st.text("")
    st.text("")
    st.text("")

    # chargement du modèle statistique 
    model = pickle.load(open(MODEL_DIR_PATH + 'lgbm.pkl',"rb"))
    
    # demande pour une table .csv en entrée 
    uploaded_file = st.file_uploader("Choose a file")
    
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file, index_col=0)
    
	# sélection d'un client par son identifiant
        idclient = st.selectbox(
        		'Quel est l\'identifiant client ?',
        		data.index)
        
        st.write('Vous avez selectionné:', idclient)
        clientdata = data.loc[[idclient]]
        
        st.text("")
    
        score = round(model.predict_proba(clientdata)[:, 1][0] * 100, 1)
    
	# affiche le score pour le client donné
        st.metric(label="Score (%)", value=score)

        st.text("")


        st.header('Données client')

        st.write(clientdata.squeeze())
