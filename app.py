import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
#streamlit supports most major plotting libraries - cool!
import pickle


st.title("Predicting Bicycle Rides")

url = r'https://raw.githubusercontent.com/JonathanBechtel/dat-11-15/main/Homework/Unit2/data/bikeshare.csv'

num_rows = st.sidebar.number_input('Select # of Rows to Load', 
                                    min_value = 1000, 
                                    max_value = 10000, 
                                    step = 1000)

section = st.sidebar.radio('Choose Application Section', ['Data Explorer', 'Model Explorer'])

#cache this expensive step so it doesn't slow us down every time
#decorators are commands you can call that influences the function below it 
@st.cache #this is a decorator
def load_data(num_rows):
    df = pd.read_csv(url, parse_dates = ['datetime'],nrows = num_rows)
    df['month'] = df['datetime'].dt.month
    df['year'] = df['datetime'].dt.year
    df['week'] = df['datetime'].dt.isocalendar().week
    df['hour'] = df['datetime'].dt.hour
    df['yesterday'] = df.groupby(['year', 'month'])['count'].shift()
    return df


#creating another function w/o a ton of logic so we can cache it 
@st.cache
def create_grouping(x_axis, y_axis):
    grouping = df.groupby(x_axis)[y_axis].mean()
    return grouping

def load_model():
    with open('pipe.pkl', 'rb') as pickled_mod:
        model = pickle.load(pickled_mod)
    return model

df= load_data(num_rows)

if section == 'Data Explorer':
    #using a pandas method that returns all categorical variables
    x_axis = st.sidebar.selectbox("Choose column for x-axis", 
                                df.select_dtypes(include = np.object).columns.tolist())
    y_axis = df['count']

    chart_type = st.sidebar.selectbox("Choose a chart Type", ['line', 'bar', 'area'])

    #st.line_chart(grouping)
    if chart_type == 'line':
        grouping = create_grouping(x_axis, y_axis)
        st.line_chart(grouping)
    elif chart_type == 'bar':
        grouping = create_grouping(x_axis, y_axis)
        st.bar_chart(grouping)
    elif chart_type == 'area':
        fig = px.strip(df[[x_axis, y_axis]], x = x_axis, y = y_axis)
        st.plotly_chart(fig)
    st.write(df)
else:
    st.text("Choose a section to explore the data")
    model = load_model()

    id_val = st.sidebar.selectbox("Choose a restaurant ID", 
                                df['id'].unique().tolist())
    yesterday = st.sidebar.number_input("How many rides yesterday?", min_value=0,
                                        max_value = 1000, step = 10, value = 190)
    day_of_week = st.sidebar.selectbox("Day of Week", 
                                        df['day_of_week'].unique().tolist())
    sample = {
    'id': id_val,
    'yesterday': yesterday,
    'day_of_week': day_of_week
    }

    sample = pd.DataFrame(sample, index=[0])
    prediction = model.predict(sample)[0]

    st.title(f"Predicted Attendance: {int(prediction)}")
    #surfacing the live prediction!!!
print(num_rows)
print(section)

##Notes
#Streamlit re-runs all code anytime something is changed
#This is why caching with decorators becomes important 
#