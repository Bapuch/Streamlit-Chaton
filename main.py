
import streamlit as st
import os
import pandas as pd
import io
from pipeline import CustomPipeline
import SessionState
import matplotlib.pyplot as plt

st.set_option('deprecation.showfileUploaderEncoding', False)

st.title('Comments')


@st.cache
def load_data(file_buffer):
    uploaded_file = io.TextIOWrapper(file_buffer)
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        return data


def rename_column(df, old_name, new_name,):
    df.rename(columns = {old_name:new_name}, inplace = True)
    st.write(f'Renaming <{old_name}> to <{new_name}>, Done!')

@st.cache
def get_proportions(data):
    sentiment_count = data.groupby(data.columns[1]).count()
    proportions = {
        sentiment_count.index[0]:{
            'count':sentiment_count.values[0][0],
            'percent': sentiment_count.values[0][0]/data.shape[0]
            },
        sentiment_count.index[1]:{
            'count':sentiment_count.values[1][0],
            'percent': sentiment_count.values[1][0]/data.shape[0]
            }, 
        }

    s = ""
    for k, v in proportions.items():
        s += f"{v['count']} ({v['percent']:.1%}) {k} comments" 
        s += " / "
    return s[:-3], proportions


def show_data(data, proportions):
    st.subheader('Data')
    st.write(data)
    st.write(proportions)


file_buffer = st.file_uploader("Choose a CSV Log File...", type="csv", encoding = None)

file = st.text_input("Choose a CSV Log File...")

session_state = SessionState.get(trained=False, pipe="", cm='', clf_report='') 

if file_buffer:
    data_load_state = st.text('Loading data...')
    data = load_data(file_buffer)
    data_load_state.text('Loading data...done!')

    str_porportions, proportions = get_proportions(data)

    fig, ax = plt.subplots()
    ax = plt.pie([proportions['Negative']['count'], proportions['Positive']['count']], labels=['Negative', 'Positive'],
    autopct='%1.1f%%', shadow=True, startangle=140)

    plt.axis('equal')
    plt.show()

    colname = st.selectbox('select a column to rename:', data.columns)
    new_colname = st.text_input("New name", "")

    

    if st.button('Rename') and new_colname!= "":

        rename_column(data, colname, new_colname)
        show_data(data, str_porportions)
        # st.plotly_chart(fig)
        st.pyplot(fig)
    else:
        show_data(data, str_porportions)
        # st.plotly_chart(fig)
        st.pyplot(fig)



    vectorizer = st.selectbox('select a vectorizer:', list(CustomPipeline.VECTORIZERS.keys()))
    model = st.selectbox('select a model:', list(CustomPipeline.MODELS.keys()))

    
    if st.button('Train'):
        session_state.pipe = CustomPipeline(vectorizer, model)
        session_state.clf_report, session_state.cm = session_state.pipe.run(data)
        session_state.trained = True

    # print("trained", session_state.trained)
    
    if session_state.trained:
        st.write(session_state.clf_report)
        # st.pyplot(cm)
        # st.write(cm)
        st.plotly_chart(session_state.cm)

        new_comment = st.text_input("Test New Comment", "")
        print(new_comment)

        if st.button('Predict') and new_comment!= "":
            print("predicting")
            prediction = session_state.pipe.predict(new_comment)
            st.success(prediction)
            st.balloons()


# st.plotly_chart(fig)
# label_input = st.text_input("Nom de la colonne de label", "label")
# comment_input = st.text_input("Nom de la colonne des commentaires", "comments")
# print(label_input)
# if st.checkbox('Show raw data'):
#     st.subheader('Raw data')
#     st.write(data)


# with st.file_input() as input:
#   if input == None:
#     st.warning('No file selected.')
#   else:
#     file_contents = input.read()