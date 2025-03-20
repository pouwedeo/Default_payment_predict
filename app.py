import streamlit as st
from src.model_deployement.modelDeployent import model_pred

# Side Bar
model_predict = st.sidebar.radio(
    "Choisir Son Modèle De Prédiction",
    ["Régression Logistique", "Arbre de Décision", "Forêt Aléatoire",
     "XgBoost", "Lightgbm", "CatBoost"]
)
# Model of predition
model = ""
if model_predict == "Régression Logistique":
    model = "src/artifacts/Logistic_predict.pkl"

elif model_predict == "Arbre de Décision":
    model = "src/artifacts/DecisionTree_predict.pkl"

elif model_predict == "Forêt Aléatoire":
    model = "src/artifacts/RandomForest_predict.pkl"

elif model_predict == "XgBoost":
    model = "src/artifacts/XGB_predict.pkl"

elif model_predict == "Lightgbm":
    model = "src/artifacts/Lgb_predict.pkl"

elif model_predict == "CatBoost":
    model = "src/artifacts/CatBoost_predict.pkl"

# Page header
st.header("Bienvenue dans votre similateur de prêt")

st.markdown("""<h5 style='color: black;'> 
   Veuillez renseigner les champs<h5>""", unsafe_allow_html=True)

# Forms
col1, col2, col3 = st.columns(3)
col4, col5, col6 = st.columns(3)

credit_lines_outstanding = col1.number_input("Lignes de crédit en cours")
loan_amt_outstanding = col2.number_input("Montant du prêt en cours")
total_debt_outstanding = col3.number_input("Dette totale en cours")

income = col4.number_input("Salaire ")
years_employed = col5.number_input("Années d'emploi")
fico_score = col6.number_input("Score de crédit")

# Feature array
features = [credit_lines_outstanding, loan_amt_outstanding,
            total_debt_outstanding, income, years_employed, fico_score]

# Prediction validation
if st.button("Valider"):
    prediction = model_pred(features, model)
    if prediction[0] == 1:
        st.markdown("""
        <p style='font-weight: bold; text-align: center; background-color:
        #c65e2df0; color: white; padding: 10px; border-radius: 5px;'>
        Désolé, votre client n'est pas éligible pour un prêt
        </p> """, unsafe_allow_html=True)
    elif prediction[0] == 0:
        st.markdown("""
        <p style='font-weight: bold; text-align: center; background-color: 
        #1ba03b; color: white; padding: 10px; border-radius: 5px;'>
         Félicitation, votre client est éligible pour un prêt!
        </p> """, unsafe_allow_html=True)


# Response
