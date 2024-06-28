import streamlit as st
from PIL import Image


def home_page():

    st.image("logo1.png",width=150)

    st.title("Football Match Outcome Predictor ⚽")

    st.write("This web app predicts the outcome of football matches in the top European leagues like: Bundesliga, EPL, La Liga, Serie A")

    st.info("While navigating through different leagues kindly please refresh the site for better performance")

    st.write("Select the league from the sidebar to get started")

    container = st.container()

    with container:
        st.write("* Bundesliga: German Football League")
        st.text("")
        st.write("* EPL: English Premier League")
        st.text("")
        st.write("* La Liga: Spanish Football League")
        st.text("")
        st.write("* Serie A: Italian Football League")
        st.text("")

    st.info("The model has been trained on 25 years of historical results (1999-2024). It makes predictions based on past encounters between the teams and their current form. Please note that these predictions are not guaranteed to be accurate and should be used as a guide rather than a definitive forecast. Factors not accounted for by the model can influence match outcomes.The predictions should not be used for betting or gambling purposes.")

    st.success("Thanks for visiting 🤩!!")

if __name__ == "__main__":
    home_page()
