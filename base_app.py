"""

    Simple Streamlit webserver application for serving developed classification
	models.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within this directory for guidance on how to use this script
    correctly.
    ---------------------------------------------------------------------

    Description: This file is used to launch a minimal streamlit web
	application. You are expected to extend the functionality of this script
	as part of your predict project.

	For further help with the Streamlit framework, see:

	https://docs.streamlit.io/en/latest/

"""
# Streamlit dependencies
import streamlit as st
import joblib
import os
from PIL import Image

# Data dependencies
import pandas as pd

# Vectorizer
news_vectorizer = open("resources/gridsearch_vectorizer.pkl", "rb")
# loading your vectorizer from the pkl file
tweet_cv = joblib.load(news_vectorizer)

# Load your raw data
raw = pd.read_csv("resources/train.csv")


def model_prediction(model_name, vect_text):
    # Load your .pkl file with the model of your choice + make predictions
    # Try loading in multiple models to give the user a choice
    predictor = joblib.load(
        open(os.path.join("resources/"+model_name+".pkl"), "rb"))
    prediction = predictor.predict(vect_text)
    message_dictionary = {
        '[-1]': 'anti', '[0]': 'a neutral message on', '[1]': 'pro', '[2]': 'news about'}
    # When model has successfully run, will print prediction
    # You can use a dictionary or similar structure to make this output
    # more human interpretable.
    model_result = "The text entered above has been classified as **" + \
        message_dictionary["{}".format(prediction)] + "** climate change."
    return model_result


# The main function where we will build the actual app

def main():
    """Tweet Classifier App with Streamlit """

    col1, col2, col3 = st.columns(3)

    with col1:
        st.image("logo/tweet.jpeg", width=200)
    with col2:
        st.write(' ')
    with col3:
        st.image("logo/climate.jpeg", width=200)

    # Creates a main title and subheader on your page -
    # these are static across all pages
    st.title("Team GridSearch AI Tweet Classifer")
    st.subheader("A Study of People's Sentiment on Climate Change")

    # Creating sidebar with selection box -
    # you can create multiple pages this way
    options = ["About Us", "Team", "Project Description",
               "Information", "Model Predictions"]
    selection = st.sidebar.selectbox("Choose Option", options)
# [theme]
# base="light"
# primaryColor="#f39241"
# textColor="#173c56"

    # Building out the "Information" page
    if selection == "Information":
        st.info(
            "Below is the raw twitter data and the class descriptions of each label")
        # You can read a markdown file from supporting resources folder
        st.markdown(""" 
Below is a description of the possible classifications and what each classification means

* **[-1] - Anti: The tweet does not believe in man-made climate change Variable definitions**

* **[0] - Neutral: The tweet neither supports nor refutes the belief of man-made climate change**

* **[1] - Pro: The tweet supports the belief of man-made climate change**

* **[2] - News: The tweet links to factual news about climate change**

""")

        st.subheader("Raw Twitter data and label")
        if st.checkbox('Show raw data'):  # data is hidden if box is unchecked
            # will write the df to the page
            st.write(raw[['sentiment', 'message']])

    # Building out the Prediction page
    if selection == "Model Predictions":
        st.info("Prediction with ML Models")
        st.markdown(
            """ You may test the 3 models' (LinearSVC, Logistic Regression & Ridge Classifer) 
            classification results by entering text in the text area below and clicking 
            the tab for which you'd like to view the results.""")

        # Creating a text box for user input
        tweet_text = st.text_area("Enter Text", "Type Here")

        # Transforming user input with vectorizer
        vect_text = tweet_cv.transform([tweet_text]).toarray()

        tab_logreg, tab_lsvc, tab3_ridge = st.tabs(
            ["LinearSVC", "Logistic Regression", "Ridge Classifier"])

        with tab_lsvc:
            st.markdown(
                """ Here a LinearSVC model is used to classify tweets into different sentiment classes.""")
            model_name = "gridsearch_final_lsvc"
            st.success(model_prediction(model_name, vect_text))
        with tab_logreg:
            st.markdown(
                """ Here a logistic regression model is used to classify tweets into different sentiment classes.""")
            model_name = "gridsearch_logistic_regression"
            st.success(model_prediction(model_name, vect_text))
        with tab3_ridge:
            st.markdown(
                """ Here a ridge classifier model is used to classify tweets into different sentiment classes.""")
            model_name = "gridsearch_ridgeclfr"
            st.success(model_prediction(model_name, vect_text))

        # if st.button("Classify"):

        # Building out the "Project Description" page
    if selection == "Project Description":
        st.info("More Information about the project")
        # You can read a markdown file from supporting resources folder
        st.markdown("""
		Many companies are built around lessening one's environmental impact or carbon
		footprint. They offer products and services that are environmentally friendly and
		sustainable, in line with their values and ideals. They would like to determine
		how people perceive climate change and whether or not they believe it is a real
		threat. This would add to their market research efforts in gauging how their
		product/service may be received.

		With this context, EDSA is challenging you during the Classification Sprint with
		the task of creating a Machine Learning model that is able to classify whether or
		not a person believes in climate change, based on their novel tweet data.

		Providing an accurate and robust solution to this task gives companies access to a
		broad base of consumer sentiment, spanning multiple demographic and geographic
		categories - thus increasing their insights and informing future marketing
		strategies.
		""")

        # Building out the predication page
    if selection == "About Us":
        st.info(
            "Below is detailed information about the our business and the GridSearch AI team")
        st.markdown("""
        GridSearch AI supports businesses to derive meaningful insight from data
        """)
        st.markdown("- GridSearch AI is a team of experienced professionals")
        st.markdown("- The team has over 6 years track record")
        st.markdown("- The business has served over 3000 customers globally")
        st.markdown("- The business leverage robust technology")

        st.markdown('''
        <style>
        [data-testid="stMarkdownContainer"] ul{
            list-style-position: inside;
        }
        </style>
        ''', unsafe_allow_html=True)

    if selection == "Team":
        st.info(
            "Below is information about the GridSearch AI team")
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
            ["Martha", "Nnaemeka", "Thepe", "Harfsa", "Orise", "Karabo"])

        with tab1:
            st.subheader("Martha Mwaura")
            st.markdown("Project Manager")
            st.image("team/martha.jpeg", width=200)
        with tab2:
            st.subheader("Nnaemeka Onyebueke")
            st.markdown("Technical Lead")
            st.image("team/nnaemeka.jpeg", width=200)
        with tab3:
            st.subheader("Thepe Mashala")
            st.markdown("Application Developer/Cloud Engineer")
            st.image("team/8324.jpg", width=200)
        with tab4:
            st.subheader("Hafsa Shariff Abass")
            st.markdown("Business Analyst")
            st.image("team/hafsa.jpeg", width=200)
        with tab5:
            st.subheader("Orisemeke Ibude")
            st.markdown("Data Scientist")
            st.image("team/orise.jpeg", width=200)
        with tab6:
            st.subheader("Karabo Eugene Hlahla")
            st.markdown("Data Engineer")
            st.image("team/karabo.jpeg", width=200)


# Required to let Streamlit instantiate our web app.
if __name__ == '__main__':
    main()
