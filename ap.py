import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import base64

def add_background_image(image_file):
    with open(image_file, "rb") as image:
        encoded = base64.b64encode(image.read()).decode()

    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
        }}

        [data-testid="stSidebar"] {{
            background-color: rgba(255, 255, 255, 0.85);
        }}

        .block-container {{
            background-color: rgba(255, 255, 255, 0.85);
            padding: 2rem;
            border-radius: 12px;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )


plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.labelsize": 11
})

st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="wide"
)

add_background_image("background.jpg")

with open("model_pickle", "rb") as file:
    model = pickle.load(file)


st.title(" Student Performance Prediction")
st.write("Predict whether a student will achieve **Good Performance (G3 ≥ 10)**.")


st.sidebar.header(" Student Features")

categorical_mappings = {
    "school": {"GP": 0, "MS": 1},
    "sex": {"F": 0, "M": 1},
    "address": {"U": 0, "R": 1},
    "famsize": {"LE3": 0, "GT3": 1},
    "Pstatus": {"A": 0, "T": 1},
    "schoolsup": {"no": 0, "yes": 1},
    "famsup": {"no": 0, "yes": 1},
    "paid": {"no": 0, "yes": 1},
    "activities": {"no": 0, "yes": 1},
    "nursery": {"no": 0, "yes": 1},
    "higher": {"no": 0, "yes": 1},
    "internet": {"no": 0, "yes": 1},
    "romantic": {"no": 0, "yes": 1},
    "Mjob": {"at_home": 0, "health": 1, "other": 2, "services": 3, "teacher": 4},
    "Fjob": {"at_home": 0, "health": 1, "other": 2, "services": 3, "teacher": 4},
    "reason": {"course": 0, "home": 1, "other": 2, "reputation": 3},
    "guardian": {"father": 0, "mother": 1, "other": 2}
}

def user_input_features():
    data = {
        'school': st.sidebar.selectbox('School', ['GP', 'MS']),
        'sex': st.sidebar.selectbox('Sex', ['F', 'M']),
        'age': st.sidebar.slider('Age', 15, 22, 17),
        'address': st.sidebar.selectbox('Address', ['U', 'R']),
        'famsize': st.sidebar.selectbox('Family Size', ['LE3', 'GT3']),
        'Pstatus': st.sidebar.selectbox('Parents Status', ['T', 'A']),
        'Medu': st.sidebar.slider('Mother Education', 0, 4, 2),
        'Fedu': st.sidebar.slider('Father Education', 0, 4, 2),
        'Mjob': st.sidebar.selectbox('Mother Job', list(categorical_mappings["Mjob"].keys())),
        'Fjob': st.sidebar.selectbox('Father Job', list(categorical_mappings["Fjob"].keys())),
        'reason': st.sidebar.selectbox('School Choice Reason', list(categorical_mappings["reason"].keys())),
        'guardian': st.sidebar.selectbox('Guardian', list(categorical_mappings["guardian"].keys())),
        'traveltime': st.sidebar.slider('Travel Time', 1, 4, 2),
        'studytime': st.sidebar.slider('Study Time', 1, 4, 2),
        'failures': st.sidebar.slider('Failures', 0, 3, 0),
        'schoolsup': st.sidebar.selectbox('School Support', ['yes', 'no']),
        'famsup': st.sidebar.selectbox('Family Support', ['yes', 'no']),
        'paid': st.sidebar.selectbox('Paid Classes', ['yes', 'no']),
        'activities': st.sidebar.selectbox('Activities', ['yes', 'no']),
        'nursery': st.sidebar.selectbox('Nursery', ['yes', 'no']),
        'higher': st.sidebar.selectbox('Higher Education', ['yes', 'no']),
        'internet': st.sidebar.selectbox('Internet', ['yes', 'no']),
        'romantic': st.sidebar.selectbox('Romantic', ['yes', 'no']),
        'famrel': st.sidebar.slider('Family Relationship', 1, 5, 4),
        'freetime': st.sidebar.slider('Free Time', 1, 5, 3),
        'goout': st.sidebar.slider('Going Out', 1, 5, 3),
        'Dalc': st.sidebar.slider('Workday Alcohol', 1, 5, 1),
        'Walc': st.sidebar.slider('Weekend Alcohol', 1, 5, 1),
        'health': st.sidebar.slider('Health', 1, 5, 3),
        'absences': st.sidebar.slider('Absences', 0, 32, 5)
    }
    return pd.DataFrame(data, index=[0])

input_df = user_input_features()


for col, mapping in categorical_mappings.items():
    input_df[col] = input_df[col].map(mapping)


st.subheader(" Encoded User Input")
st.dataframe(input_df)


if st.button("Predict Performance"):

    prediction = model.predict(input_df)
    prediction_proba = model.predict_proba(input_df)

    result = "Good Performance (G3 ≥ 10)" if prediction[0] == 1 else "Poor Performance (G3 < 10)"
    st.success(result)

    col1, col2 = st.columns(2)

   
    with col1:
        st.subheader("Prediction Probability")

        fig1, ax1 = plt.subplots(figsize=(2,2))
        ax1.pie(
            prediction_proba[0],
            labels=["Poor", "Good"],
            autopct="%1.1f%%",
            startangle=90,
            colors=["#02B96D", "#FF7B00FF"],
            wedgeprops={"width": 0.4}
        )
        ax1.axis("equal")
        st.pyplot(fig1, use_container_width=True)

   
    with col2:
        st.subheader("User Numeric Feature Summary")

        numeric_features = [
            'age', 'studytime', 'failures',
            'famrel', 'freetime', 'goout',
            'Dalc', 'Walc', 'health', 'absences'
        ]

        values = input_df[numeric_features].iloc[0]

        fig2, ax2 = plt.subplots(figsize=(10, 5))
        ax2.bar(values.index, values.values, color="#000000")
        ax2.set_ylabel("Value")
        ax2.set_title("User Provided Feature Values")
        ax2.set_xticks(range(len(values)))
        ax2.set_xticklabels(values.index, rotation=45)

        st.pyplot(fig2)


numeric_features = [
    'age', 'studytime', 'failures',
    'famrel', 'freetime', 'goout',
    'Dalc', 'Walc', 'health', 'absences'
]

# Take the first row (user input)
values = input_df[numeric_features].iloc[0]

# Create line chart
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(values.index, values.values, marker='o', linestyle='-', color='#00C1FC')
ax.set_title("User Numeric Features Trend", fontsize=16)
ax.set_ylabel("Value", fontsize=12)
ax.set_xlabel("Features", fontsize=12)
ax.grid(True, linestyle='--', alpha=0.5)
ax.set_xticks(range(len(values)))
ax.set_xticklabels(values.index, rotation=45)

st.subheader("Line Chart (Matplotlib)")
st.pyplot(fig, use_container_width=True)