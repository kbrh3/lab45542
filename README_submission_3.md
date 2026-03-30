# README Submission 3: PolicyPulse

Hi! Welcome to **PolicyPulse**. I've been working on this project as I learn more about building AI applications. This README explains what the project is, how it works, and how to get it running. 

## 1. Project Overview

**PolicyPulse** is an AI-powered system designed to answer questions about legislation. You can ask it questions about different bills or laws, and it will try to find the right information to give you a clear answer. It's essentially a smart search engine specifically for legislative data!

## 2. How the System Works

The system is built using three main pieces that talk to each other:
- **The Frontend (Streamlit):** This is the user interface where you type your questions. It's simple and easy to use.
- **The Backend (FastAPI):** This is the "brain" of the app. It takes your question from the frontend, figures out what to do with it, and searches for the answer.
- **The Database (Snowflake):** This is where all the legislative data lives. Instead of looking through messy local files, the backend speaks to Snowflake to retrieve the exact text of the bills needed to answer your question.

## 3. What Was Fixed and Improved

When I first started jumping into the code, the system was a bit messy. It originally had issues where it returned incorrect data from old, outdated PDF files. Here is how I helped fix and improve it:

- **Upgraded to Snowflake:** I removed the old PDF-based data entirely. Now, the system uses a Snowflake-based retrieval system, which is much more structured and accurate.
- **Traced the Backend:** I spent time tracing the real execution path of the backend so I could see exactly how data was moving through the app and where things were getting stuck.
- **Added Debugging Tools:** I added tools to help monitor what the system is doing under the hood, making it easier to catch errors during retrieval.
- **Improved the UI:** I made the Streamlit frontend look and feel a lot nicer!
- **Added a Safe Fallback:** The system used to try and guess answers even if it found nothing. I added a safe fallback feature so that if no relevant legislative evidence is found, it will just honestly tell you that instead of making things up.

## 4. How to Run the Project

If you want to run this on your own computer, here are the basic steps:

1. **Open the project folder** in your terminal.
2. **Set up your environment:** Make sure you have your Python virtual environment activated. Create a `.env` file (you can copy `.env.example` if it's there) and add your Snowflake credentials and any necessary API keys.
3. **Start the Backend:** Open a terminal window and run:
   ```bash
   uvicorn api.server:app --host 0.0.0.0 --port 8000
   ```
4. **Start the Frontend:** Open a *second* terminal window and run:
   ```bash
   streamlit run app/main.py
   ```
5. **Open your browser** to `http://localhost:8501` to start asking questions!

## 5. Known Issues

- **Limited Data:** Right now, there isn't a massive amount of data in the Snowflake database yet. Because of this, the AI might not know the answer to every obscure legislative question you throw at it.

## 6. What Still Needs to be Improved

Moving forward, there are a few things that still need to be improved:
- **Adding More Data:** The biggest priority is loading more bills into Snowflake so the system can answer a wider variety of questions.
- **Faster Retrieval:** As the database grows, we'll need to figure out ways to make sure the app stays fast and responsive. 
- **Better Handling of Complex Questions:** Sometimes users ask multi-part questions, and the system could do a better job of breaking those down before searching the database.

Thanks for checking out the project!
