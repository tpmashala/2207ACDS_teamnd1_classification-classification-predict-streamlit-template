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
news_vectorizer = open("resources/tfidfvect.pkl", "rb")
# loading your vectorizer from the pkl file
tweet_cv = joblib.load(news_vectorizer)

# Load your raw data
raw = pd.read_csv("resources/train.csv")

# The main function where we will build the actual app


def main():
    """Tweet Classifier App with Streamlit """

    # Creates a main title and subheader on your page -
    # these are static across all pages
    st.title("Team GridSearch AI Tweet Classifer")
    st.subheader("Climate change tweet classification")

    # Creating sidebar with selection box -
    # you can create multiple pages this way
    options = ["Prediction", "Information", "Project Description", "About Us"]
    selection = st.sidebar.selectbox("Choose Option", options)

    # Building out the "Information" page
    if selection == "Information":
        st.info("General Information")
        # You can read a markdown file from supporting resources folder
        st.markdown(""" ***Class Description***

Below is a description of the possible responses/classifcations from running the model
and what each classification means

2 News: the tweet links to factual news about climate change

1 Pro: the tweet supports the belief of man-made climate change

0 Neutral: the tweet neither supports nor refutes the belief of man-made climate change

-1 Anti: the tweet does not believe in man-made climate change Variable definitions

""")

        st.subheader("Raw Twitter data and label")
        if st.checkbox('Show raw data'):  # data is hidden if box is unchecked
            # will write the df to the page
            st.write(raw[['sentiment', 'message']])

    # Building out the Prediction page
    if selection == "Prediction":
        st.info("Prediction with ML Models")
        st.markdown(""" The project uses a logistic regression model to classify tweets
		into different sentiment classes.""")
        # Creating a text box for user input
        tweet_text = st.text_area("Enter Text", "Type Here")

        if st.button("Classify"):
            # Transforming user input with vectorizer
            vect_text = tweet_cv.transform([tweet_text]).toarray()
            # Load your .pkl file with the model of your choice + make predictions
            # Try loading in multiple models to give the user a choice
            predictor = joblib.load(
                open(os.path.join("resources/Logistic_regression.pkl"), "rb"))
            prediction = predictor.predict(vect_text)

            # When model has successfully run, will print prediction
            # You can use a dictionary or similar structure to make this output
            # more human interpretable.
            st.success("Text Categorized as: {}".format(prediction))

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
        st.info("Team Profile")
        st.markdown("""The tabs below contain detailed information about 
		each team member who was involved in the project""")

        tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(
            ["Martha", "Nnaemeka", "Thepe", "Harfsa", "Orise", "Karabo", "Koketso"])

        with tab1:
            st.header("Martha Mwaura")
            st.image("team/martha.jpeg", width=200)
        with tab2:
            st.header("Nnaemeka Onyebueke")
            st.image("team/nnaemeka.jpeg", width=200)
        with tab3:
            st.header("Thepe Mashala")
            st.image("team/8324.jpg", width=200)
        with tab4:
            st.header("Hafsa Shariff Abass")
            st.image("team/hafsa.jpeg", width=200)
        with tab5:
            st.header("Orisemeke Ibude")
            st.image("team/orise.jpeg", width=200)
        with tab6:
            st.header("Karabo Eugene Hlahla")
            st.image("https://static.streamlit.io/examples/owl.jpg", width=200)
        with tab7:
            st.header("Koketso Maleka")
            st.image("https://static.streamlit.io/examples/owl.jpg", width=200)


# Required to let Streamlit instantiate our web app.
if __name__ == '__main__':
    main()
