import streamlit as st 

st.sidebar.radio("Regression Logistique", 2)
st.header("Bienvenue dans votre similateur de prêt")

st.markdown("""
   <h5 style='color: purple; text-decoration: underline;'> 
   Veuillez Renseigner Les Champs<h5>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

col4, col5, col6 = st.columns(3)

col1.number_input("age")
col2.number_input("prix")
col3.number_input("sexe")

col4.number_input("nom")
col5.number_input("fix")
col6.number_input("emal")

st.button("Valider")


st.markdown("""
      <p style='font-weight: bold; text-align: center; background-color: #188632; color: white; padding: 10px; border-radius: 5px;'>
      Bonjour
       </p>
    """, unsafe_allow_html=True)