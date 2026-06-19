# Acko-Insurance-AI-Platform
Customers talk to an AI assistant that understands their insurance questions and answers from real Acko policy documents.
Problem Statement
What is Acko Insurance?
Acko Insurance is India's first fully digital insurance company. Unlike traditional insurers,
Acko has no physical branches or agents — everything happens online. They cover cars,
bikes, health, and travel, serving over 2.8 crore customers across India.
What Problem Are We Solving?
Today, when a customer wants to:
• Ask a simple question like "Does my insurance cover flood damage?" — they have
to read long PDF documents or wait in a call centre queue.
• Get an insurance quote — they fill a form, wait for an agent to call back, and often
receive a vague number.
• File a claim after an accident — they submit paperwork, wait days for a human to
review the damage, and have no idea when or how much they will receive.
At the same time, Acko's internal managers have no easy way to ask simple business
questions like "How many bike claims came in this week?" without going to a data team.
What Will This Project Build?
This project builds a smart, chat-first web application where:
• Customers talk to an AI assistant that understands their insurance questions and
answers from real Acko policy documents.
• Customers can get an instant premium quote just by describing their vehicle — no
agent needed.

• Managers can view a live analytics dashboard and ask questions about their data in
plain English.
Module 1 — AI Policy Chatbot
Simple Explanation:
Think of this as a smart customer service agent that has read every Acko insurance
document and can answer any question a customer asks — instantly, accurately, and in
simple language.
"Does my bike insurance cover theft if I forgot to lock it?"
The chatbot finds the right section in Acko's policy documents and replies in plain language
— no jargon, no waiting.

What the Student Builds
• A chat interface built with Streamlit where customers can type questions.
• A knowledge base from Acko's actual policy PDFs — motor, health, travel — broken
into small searchable pieces.
• ChromaDB stores those pieces as searchable vectors.
• Gemini reads the most relevant pieces and writes a clear, helpful answer.
• Every question and answer is saved to the database so managers can see what
customers ask most.

How It Works — Step by Step
1. Customer types a question in the chat box.
2. The LangChain / LangGraph agent receives the question and understands what the
customer is asking.
3. ChromaDB searches through all Acko policy document chunks and finds the 3–5
most relevant paragraphs.
4. Those paragraphs are sent to Gemini (free API) as context.
5. Gemini reads the context and writes a clear, friendly answer for the customer.
6. The answer appears in the chat. The conversation is logged to PostgreSQL.

   Tools and Technologies
What It Does Tool Used
AI Orchestration (the brain) LangChain or LangGraph
AI that writes the answer Google Gemini API (Free tier)
Document search engine ChromaDB — stores and searches policy document chunks
Loading and chunking PDFs LangChain Document Loaders +
RecursiveCharacterTextSplitter

Text-to-vector conversion Google Generative AI Embeddings (free) or HuggingFace

sentence-transformers

User interface (chat) Streamlit — st.chat_input, st.chat_message
Saving conversations PostgreSQL + SQLAlchemy ORM

Module 2 — Insurance Premium Quote Predictor
Simple Explanation:
When a customer asks "How much will insurance cost for my 2020 Honda City in
Chennai?", this module gives them an instant, accurate price estimate — calculated by a
Machine Learning model trained on thousands of similar insurance cases. No agent call.
No waiting.

What the Student Builds
• A simple form inside the chatbot where the customer enters vehicle details.
• A feature engineering pipeline that converts raw inputs into meaningful signals the
ML model understands.
• A trained model that predicts the annual premium in Rupees.
• A result display showing the predicted price, and a chart explaining which factors
affected the price the most (vehicle age, city, IDV).
• All quotations are saved to the database for management reporting.

What Information the Customer Provides
• Vehicle type (Car or Bike)
• Vehicle make and model (e.g., Maruti Swift, Honda Activa)
• Year of manufacture
• Fuel type (Petrol / Diesel / Electric)
• City of registration
• IDV — Insured Declared Value (the current market value of the vehicle)
• No Claim Bonus (NCB) — discount earned for safe driving

Tools and Technologies
What It Does Tool Used
Premium prediction model Regressor (Scikit-learn)
Explaining the prediction SHAP — shows which factors drove the premium up or down
Data preparation Pandas, Scikit-learn feature engineering pipeline
Saving model file joblib (.pkl file stored on AWS S3)
User interface Streamlit form within the chatbot interface
Storing quotations PostgreSQL via SQLAlchemy
Module 4 — Management Analytics Dashboard
Simple Explanation:
A live visual control panel for Acko's managers — all the important business numbers in
one place, updating in real time from the database.

What the Manager Sees
• Total number of claims submitted today, this week, and this month.
• Average predicted claim payout — for cars and bikes separately.
• Total insurance quotations generated and the average premium quoted.
• Top cities by claim volume — shown on a bar chart.
• Claim volume trend over time — shown as a line chart.
• Top 10 questions customers asked the chatbot this week.

Tools and Technologies
What It Does Tool Used
KPI cards (big numbers) Streamlit st.metric, st.columns
Bar and line charts Plotly Express
Trend charts Altair
Data source PostgreSQL (AWS RDS) — live queries on every refresh
Access control Streamlit session state — managers log in with a password

Module 5 — Manager AI Assistant (Ask in Plain English)
Simple Explanation:
Module 4 shows fixed charts. Module 5 lets the manager ask any question they want — in
plain English — and get an answer from the live database. No SQL. No data team. Instant
answer.

Example Questions a Manager Can Ask

Manager Asks... What the AI Does

"How many car claims were filed this month?" Queries claims table → Counts records →

Returns the number

"What is the average payout for bike claims in
Chennai?"

Filters by city and vehicle type → Returns
average claim amount

"Which vehicle model has the most claims this
year?"

Groups by model → Counts → Returns top
model with count

"Show me claims with approval probability
below 40%"

Filters claims table → Returns list for human
review

"What are the top 5 questions customers asked
this week?"

Queries chat_logs → Groups by question →
Returns ranked list

"How many quotations did we generate for
electric cars this quarter?"

Filters quotations by fuel type and date →
Returns count

Tools and Technologies
What It Does Tool Used
AI agent that runs the
conversation

LangGraph — stateful agent with memory across turns

Converts question to SQL
and runs it

LangChain SQLDatabaseChain + PostgreSQL

AI that writes the final answer Gemini API — (Free tier)
Live data source PostgreSQL (AWS RDS)
Manager interface Streamlit chat interface — role-gated to managers only

Cloud Deployment
• Streamlit app is publicly accessible via AWS EC2 on port 8501.
Project Evaluation Metrics
The following metrics will be used to evaluate the quality and correctness of the project:

Module 1 — Chatbot Quality
• Retrieval Relevance: Does ChromaDB return the correct policy chunk for a given
question? Test with 20 Q&A pairs.
• Answer Accuracy: Does Gemini's response correctly reflect the retrieved policy
content?
• Response Time: Answer delivered in under 5 seconds for standard queries.
Module 2 — Quotation Model
• R2 Score: Above 0.85 on the held-out test set — measures how well the model
explains premium variation.

• RMSE: Root Mean Squared Error within ₹2,000 of actual premium — practical
accuracy.
• SHAP Correctness: Top features (IDV, city, vehicle age) align with domain
knowledge.
Module 4 — Dashboard
• Data Accuracy: Dashboard numbers match the actual records in PostgreSQL.
• Chart Clarity: Visualisations correctly represent claim volume, city distribution, and
approval rates.
• Refresh Correctness: Dashboard reflects newly submitted data within one page
refresh.
Module 5 — Manager AI Assistant
• SQL Correctness: Generated SQL queries return the correct result for at least 8 out
of 10 test questions.
• Answer Quality: Gemini response is clear, uses actual numbers from the database,
and is not hallucinated.
Overall System
• End-to-End Flow: Customer question → RAG agent → correct tool → response →
database write → dashboard update all work without errors.
• AWS Deployment: App is accessible publicly, S3 uploads work, and RDS is
connected.

Technical Tags
Python, Streamlit, LangChain, LangGraph, ChromaDB, Google Gemini API (Free), Gemini
Vision, RAG (Retrieval-Augmented Generation), Conversational AI, Agentic AI, , Scikit-learn,
SHAP, Feature Engineering, Computer Vision, OCR, AWS EC2, AWS S3, AWS RDS,
PostgreSQL, SQLAlchemy, Plotly, Altair, NL to SQL, InsurTech, End-to-End AI
System Design, Cloud Deployment
