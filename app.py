import streamlit as st
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

st.set_page_config(page_title="SaaS Conversion Predictor", layout="wide")
st.title("Micro-SaaS Conversion Predictor")
st.write("This app predicts the probability of a user converting to a paid plan based on various features.")    

@st.cache_data
def load_and_train_model():
    df = pd.read_csv('saas_data.csv')
    X = df.drop('purchased', axis=1)
    y = df['purchased']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(X_scaled, y)


    return model, scaler, df, X_scaled, y

model, scaler, df, X_scaled, y = load_and_train_model()

st.sidebar.header("Test a New User")

user_mins = st.sidebar.slider("Minutes spent on the platform", 0, 300, 60)
user_logins = st.sidebar.slider("Number of logins", 0, 50, 5)

user_data_scaled = scaler.transform([[user_mins, user_logins]])
prediction = model.predict(user_data_scaled)[0]

st.subheader("Prediction")
if prediction == 1:
    st.success("🟢 **High Intent:** This user behaves like paying customers. Offer a premium discount!")
else:
    st.error("🔴 **Low Intent:** This user behaves like free-tier users. Send an engagement email.")


#Plotting the graph of the decision boundary and the user's position

st.subheader("KNN Visualization")
st.write("The star represents the new user. KNN looks at the 5 closest dots to the star to make its prediction.")

fig, ax = plt.subplots(figsize=(8, 5))

# Plot the training data
scatter = ax.scatter(
    df['session_minutes'], 
    df['login_count'], 
    c=df['purchased'], 
    cmap='coolwarm', 
    alpha=0.6, 
    edgecolors='k'
)

# Plot the new user
ax.scatter(
    user_mins, 
    user_logins, 
    color='gold', 
    marker='*', 
    s=400, 
    edgecolors='black', 
    label='New User'
)

ax.set_xlabel("Session Minutes")
ax.set_ylabel("Login Count")
ax.set_title("User Behavior Clusters")
ax.legend()
ax.grid(True, linestyle=':', alpha=0.6)

st.pyplot(fig)