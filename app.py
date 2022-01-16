import streamlit as st
import category_encoders as ce
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
#streamlit supports most major plotting libraries - cool!
import pickle


st.title("Predicting Bicycle Rides")

url = r'https://raw.githubusercontent.com/JonathanBechtel/dat-11-15/main/Homework/Unit2/data/bikeshare.csv'

num_rows = st.sidebar.number_input('Select # of Rows to Load', 
                                    min_value = 1000, 
                                    max_value = 100000, 
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
    df['yesterday'] = df.groupby('hour')['count'].shift()
    return df


#creating another function w/o a ton of logic so we can cache it 
#@st.cache
def create_grouping(x_axis, y_axis):
    grouping = df.groupby(x_axis)[y_axis].mean()
    return grouping

def load_model():
    with open('simple_pipe.pkl', 'rb') as pickled_mod:
        model = pickle.load(pickled_mod)
    return model

df= load_data(num_rows)

if section == 'Data Explorer':
    #using a pandas method that returns all categorical variables
    x_axis = st.sidebar.selectbox("Choose column for x-axis", 
                                df.select_dtypes(include = np.object).columns.tolist())
    y_axis = 'count'

    chart_type = st.sidebar.selectbox("Choose a chart Type", ['histogram','line', 'bar', 'area', 'trend'])

    #do a timeline chart 
    if chart_type == 'histogram':
        st.subheader('Number of rides by hour')
        fig = px.histogram(df, x="hour", y = 'count', nbins=24)
        st.plotly_chart(fig)
    elif chart_type == 'line':
        grouping = create_grouping(x_axis, y_axis)
        st.line_chart(grouping)
    elif chart_type == 'bar':
        grouping = create_grouping(x_axis, y_axis)
        st.bar_chart(grouping)
    elif chart_type == 'area':
        fig = px.strip(df[[x_axis, y_axis]], x = x_axis, y = y_axis)
        st.plotly_chart(fig)
    elif chart_type == 'trend':
        data = df.groupby(['year','month'])['count'].sum().reset_index()
        fig = px.line(data, x='month', y='count', facet_row = 'year', title='Monthly Rides Over Time')
        #fig = px.strip(df[[x_axis, y_axis]], x = x_axis, y = y_axis)
        st.plotly_chart(fig)
    st.write(df)
else:
    st.text("Let's predict how many rides will happen at a given time tomorrow")
    model = load_model()

    hour = st.sidebar.selectbox("What time of day?", 
                                df['hour'].unique().tolist())
    yesterday = st.sidebar.number_input("How many rides at this time yesterday?", min_value=0,
                                        max_value = 1000, step = 10, value = 190)
    atemp = st.sidebar.number_input("How hot will it feel tomorrow?", min_value=0,
                                        max_value = 45, step = 1, value = 24) 
    season = st.sidebar.selectbox("What season is it?", 
                                df['season'].unique().tolist())   
    workingday_query = st.sidebar.radio("Is it a working day?", 
                                        ('Yes', 'No'))
    workingday = 1
    if workingday_query == "No": 
        workingday = 0
    sample = {
    'yesterday': yesterday,
    'hour': hour,
    'atemp': atemp,
    'workingday': workingday,
    'season': season
    }

    sample = pd.DataFrame(sample, index=[0])
    st.write(sample)
    prediction = model.predict(sample)[0]

    st.title(f"Predicted Rides tomorrow at {hour}:00: {int(prediction)}")
    #surfacing the live prediction!!!
print(num_rows)
print(section)

##Notes
#Streamlit re-runs all code anytime something is changed
#This is why caching with decorators becomes important 
#