import streamlit as st
from src.model_deployement.modelDeployent import model_pred

# Side Bar
model_predict = st.sidebar.radio(
    "Choisir Son Modèle De Prédiction",
    ["Régression Logistique", "Arbre de Décision", "Forêt Aléatoire",
     "XgBoost", "Lightgbm", "CatBoost"]
)

model = ""
if model_predict == "Régression Logistique":
    model = "src/artifacts/Logistic_predict.pkl"
elif model_predict == "Arbre de Décision":
    st.write("Arbre de Décision")
elif model_predict == "Forêt Aléatoire":
    st.write("Forêt Aléatoire")

elif model_predict == "XgBoost":
    st.write("XgBoost")
elif model_predict == "Lightgbm":
    st.write("Lightgbm")
elif model_predict == "CatBoost":
    st.write("CatBoost")


st.header("Bienvenue dans votre similateur de prêt")

st.markdown("""
   <h5 style='color: black;'> 
   Veuillez Renseigner Les Champs<h5>
""", unsafe_allow_html=True)

# Forms
col1, col2, col3 = st.columns(3)

col4, col5, col6 = st.columns(3)
id = col1.number_input("id")
credit_lines_outstanding = col1.number_input("Lignes De Crédit En Cours")
loan_amt_outstanding = col2.number_input("Montant Du Prêt En Cours")
total_debt_outstanding = col3.number_input("Dette Totale En Cours")

income = col4.number_input("Salaire ")
years_employed = col5.number_input("Années D'Emploi")
fico_score = col6.number_input("Score De Crédit")

features = [id, credit_lines_outstanding, loan_amt_outstanding,
            total_debt_outstanding, income, years_employed, fico_score]

if st.button("Valider"):
    prediction = model_pred(features, model)
    if prediction[0] == 1:
        st.markdown("""
        <p style='font-weight: bold; text-align: center; background-color: #188632; color: white; padding: 10px; border-radius: 5px;'>
        Bonjour
        </p>
        """, unsafe_allow_html=True)
    elif prediction[0] == 0:
        st.markdown("""
        <p style='font-weight: bold; text-align: center; background-color: #188632; color: white; padding: 10px; border-radius: 5px;'>
        Bonsoir
        </p>
        """, unsafe_allow_html=True)


# Response
