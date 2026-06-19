import uuid
from sqlalchemy import engine
import streamlit as st
from langchain_chroma import Chroma
from db import engine 
from db import (
     log_conversation, save_quote,
     get_city_distribution,
    get_quotation_stats, SessionLocal,log_chat_interaction)
import plotly.graph_objects as go

from langchain_community.utilities.sql_database import SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain

from langchain_google_genai import ChatGoogleGenerativeAI
import os
from dotenv import load_dotenv
import pandas as pd
import joblib
import shap
import seaborn as sns
import matplotlib.pyplot as plt
import datetime
import boto3

from streamlit_option_menu import option_menu

# 1. as sidebar menu
with st.sidebar:
    selected = option_menu(
    "Main Menu",
    ["AI Policy Chatbot", "Insurance Premium Quote Predictor", "Management Analytics Dashboard and  Query AI Assistant"],
    icons=['gear', 'clipboard',"bar_chart","tram","abacus"],
    menu_icon="cast",
    default_index=1
)

if selected== "AI Policy Chatbot": 
# Load environment variables
    load_dotenv()

# Load ChromaDB
    vectordb = Chroma(persist_directory="chroma_db")

# Gemini model
    chat_model = ChatGoogleGenerativeAI(
    model='gemini-3.1-flash-lite',
    temperature=0.3,
    api_key=os.getenv("GEMINI_API_KEY")
    )

    def INDENT_Classifier(query):
        messages = [
        ("system", "Classify into [Greetings, General, Policy_Q&A]"),
        ("human", query)
        ]
        response = chat_model.invoke(messages)
        if isinstance(response.content, str):
            return response.content.strip()
        elif isinstance(response.content, list):
            return response.content[0].get("text", "")
        return ""

    def get_answer(query):
        indent = INDENT_Classifier(query)
        if indent in ["General", "Greetings"]:
            messages = [
            ("system", "You are Expert ACKO Insurance Sale Assistance. "
                       "Respond politely, greet the customer gently. "
                       "If they ask anything apart from our business scope, "
                       "don't answer general questions. Just mention your role "
                       "and what you can do."),
            ("human", query)
            ]
            response = chat_model.invoke(messages)
        else:
            result = vectordb.similarity_search(query, k=4)
            context = [i.page_content for i in result]
            messages = [
            ("system", f"You are Expert ACKO Insurance Sale Assistance. "
                       f"Answer based on context {context}, structured, ≤300 words. "
                       f"Use tables if required."),
            ("human", query)
            ]
        response = chat_model.invoke(messages)

        if isinstance(response.content, str):
            return response.content.strip()
        elif isinstance(response.content, list):
            return response.content[0].get("text", "")
        return ""

# --- Streamlit UI ---

    st.title("ACKO Insurance Assistant")

# Chatbot section
    st.subheader("Policy Chatbot")
    user_query = st.chat_input("Ask about your policy...")
    if user_query:
        try:
            answer = get_answer(user_query)
            st.chat_message("user").write(user_query)
            st.chat_message("assistant").write(answer)
            log_conversation(user_query, answer)
        except Exception as e:
            st.error(f"Error: {e}")
#############################################################################################################################
elif selected== "Insurance Premium Quote Predictor": 
# Premium Quote Predictor section
    st.subheader("Insurance Premium Quote Predictor")

    vehicle_type = st.selectbox("Vehicle Type", ["Car", "bike"])
    vehicle_make = st.text_input("Vehicle Make")
    vehicle_model = st.text_input("Vehicle Model")
    fuel_type = st.selectbox("Fuel Type", ["Petrol", "Diesel", "Electric"])
    city = st.text_input("City of Registration")
    manufacturing_year = st.number_input("Year of Manufacture", min_value=1990, max_value=2026, step=1)
    idv = st.number_input("Insured Declared Value (₹)", min_value=100, step=1000)
    ncb_percent = st.slider("NCB (%)", 0, 50, 0)
    vehicle_age = datetime.datetime.now().year - manufacturing_year

# Load model
    model = joblib.load("models/premium_model.pkl")


# Prepare input
    input_data = pd.DataFrame([{
    "vehicle_type": vehicle_type,
    "vehicle_make": vehicle_make,
    "vehicle_model": vehicle_model,
    "fuel_type": fuel_type,
    "city": city,
    "manufacturing_year": manufacturing_year,
    "vehicle_age": vehicle_age,
    "idv": idv,
    "ncb_percent": ncb_percent
    }])

    if st.button("Get Quote"):
        try:
        # Predict premium
            prediction = model.predict(input_data)[0]
            st.success(f"Estimated Premium: ₹{prediction:,.2f}")

        # Save quotation to DB
            save_quote(input_data.iloc[0].to_dict(), prediction)
        

        # SHAP explanation with seaborn
            try:
                explainer = shap.TreeExplainer(model.named_steps["regressor"])
                X_transformed = model.named_steps["preprocessor"].transform(input_data)
                shap_values = explainer(X_transformed)

                feature_names = model.named_steps["preprocessor"].get_feature_names_out()
                shap_df = pd.DataFrame({
                "feature": feature_names,
                "shap_value": shap_values.values[0]
                })

            # Filter out near-zero SHAP values and sort by absolute impact
                shap_df = shap_df[shap_df["shap_value"].abs() > 1e-6]
                shap_df = shap_df.sort_values("shap_value", key=abs, ascending=False)

                st.write("Factors influencing premium (SHAP values):")

            # Show only top 10 features for readability
                top_n = 10
                shap_df_top = shap_df.head(top_n)

            # Clean feature names for readability
                shap_df_top["feature"] = shap_df_top["feature"].str.replace("cat__", "", regex=False)
                shap_df_top["feature"] = shap_df_top["feature"].str.replace("vehicle_make_", "Make: ", regex=False)
                shap_df_top["feature"] = shap_df_top["feature"].str.replace("vehicle_type_", "Type: ", regex=False)

            # Plot SHAP values
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.barplot(data=shap_df_top, x="shap_value", y="feature", palette="coolwarm", ax=ax)
                ax.set_title(f"Top {top_n} Feature Importance (SHAP values)")
                ax.set_xlabel("SHAP Value (Impact on Premium)")
                ax.set_ylabel("Feature")
                plt.tight_layout()
                st.pyplot(fig)

            except Exception as e:
                st.error(f"Error generating SHAP explanation: {e}")

        except Exception as e:
            st.error(f"Error during prediction: {e}")


###################################################################################################
elif selected == "Management Analytics Dashboard and  Query AI Assistant":
    st.title("📊 Management Analytics Dashboard and Query AI Assistant")

    # Authentication
    if "manager_authenticated" not in st.session_state:
        pwd = st.text_input("Manager Password", type="password", key="manager_pwd")
        if st.button("Login", key="manager_login"):
            if pwd == os.getenv("MANAGER_PASSWORD", "admin123"):
                st.session_state.manager_authenticated = True
                st.success("✅ Access granted")
            else:
                st.error("❌ Incorrect password")

    if not st.session_state.get("manager_authenticated", False):
        st.stop()

    # --- Management Analytics Dashboard ---
    session = SessionLocal()

    # KPI cards
   

# City distribution
    city_counts = get_city_distribution(session)
    city_df = pd.DataFrame(city_counts.items(), columns=["City", "Count"])

# Create waterfall chart
    fig = go.Figure(go.Waterfall(
    name="City Claims",
    orientation="v",
    x=city_df["City"],
    y=city_df["Count"],
    connector={"line": {"color": "rgb(63, 63, 63)"}}
    ))

    fig.update_layout(
    title="Claims Distribution by City (Waterfall)",
    waterfallgap=0.3
    )

    st.plotly_chart(fig, use_container_width=True)


    # Quotations
    total_quotes, avg_premium = get_quotation_stats(session)
    st.metric("Total Quotations", total_quotes)
    st.metric("Avg Premium Quoted", f"{avg_premium:.2f}")



    # Setup DB + LangChain
    db = SQLDatabase(engine)
    chat_model = ChatGoogleGenerativeAI(
        model="gemini-3.1-flash-lite",
        temperature=0.3,
        api_key=os.getenv("GEMINI_API_KEY")
    )
    db_chain = SQLDatabaseChain.from_llm(chat_model, db, verbose=True)

    # --- Streamlit Chat UI ---
    st.subheader("Query AI Assistant")
    manager_query = st.chat_input("Ask in plain English...")

    if manager_query:
        st.chat_message("user").write(manager_query)
        try:
            response = db_chain.run(manager_query)
            st.chat_message("assistant").write(response)

            # Log interaction
            user_id = uuid.uuid4()  # Replace with real manager UUID if available
            log_chat_interaction(
                user_id=user_id,
                intent="policy_qa",   # adjust based on context
                question=manager_query,
                retrieved_source="db_chain"
            )
        except Exception as e:
            st.error(f"Error: {e}")



       
