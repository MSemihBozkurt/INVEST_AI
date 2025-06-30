# INVEST AI - AI Powered Investment Advisor and Analyzer

The project is a comprehensive, Gradio-based investment assistant application designed to provide personalized investment advice, market sentiment analysis, and educational guidance for investors of all levels. The application integrates multiple AI agents powered by Google's **Gemini 2.0 Flash** model, real-time financial data from Yahoo Finance and Finnhub APIs, Google Search Results from SerpApi and an advanced Retrieval-Augmented Generation (RAG) system for investment education.

## About The Project

The application begins with a detailed risk profile questionnaire to assess the user's financial situation, investment goals, knowledge level, and risk tolerance. Based on the answers, a personalized investor profile is generated.

The main chat assistant then uses this profile to:
1.  Identify suitable stock symbols for the user via an **Insight Agent**.
2.  Fetch historical price data, recent company news, and annual financial reports for these stocks using the Yahoo Finance and Finnhub APIs.
3.  Generate data-driven, justified, and tailored investment recommendations through a specialized **Recommendation Agent** that analyzes all the collected data.

Additionally, the platform includes a market analysis tool that searches for recent news on any stock or financial topic, summarizes it using AI, and provides a sentiment analysis classification of based on the content.

For novice investors, there is a guidance chatbot based on RAG technology. It answers investment-related questions by referencing a curated collection of financial documents and providing citations for its answers.

## Key Features

* **Personalized Investor Profile:** In-depth questionnaire for custom risk and goal analysis.
* **AI-Powered Chat Assistant:** Delivers tailored investment recommendations using Gemini agents.
* **Market Sentiment Analysis:** Real-time sentiment analysis based on news enriched by Google Search.
* **RAG-Based Educational Bot:** Answers investment questions with source-cited responses from a local PDF library.
* **Real-Time Data Integration:** Live stock prices, news, and financial statements from Yahoo Finance, Finnhub APIs and SerpApi.
* **User-Friendly Interface:** A multi-panel interface built with Gradio for seamless switching between the questionnaire, chat, analysis, and education modes.
* **Extra Functionalities:** API usage and cost tracking, process logging for transparency, Excel export for financial data, and chat history management.

## Technologies and Techniques Used

This project effectively utilizes **non-parametric grounding** techniques (RAG, Multi-Agent Framework, Function Calling, Structured Output), which are fundamental to modern LLM applications.

* **AI Model:** The agents and chatbots are powered by Google's **Gemini 2.0 Flash** model.
* **Grounding Techniques:**
    * **Agents:** AI agents that autonomously execute multi-step tasks such as identifying suitable stocks, gathering data, and formulating recommendations based on user input.
    * **Function Calling:** The ability of agents to interact with external APIs like Yahoo Finance, Finnhub, and Google Search (SerpAPI) to retrieve accurate, real-time data.
    * **Retrieval-Augmented Generation (RAG):** The educational bot retrieves relevant information from an external knowledge base (PDF vectors stored in ChromaDB) to ground its answers, making them more accurate and reliable.
    * **Structured Output:** The Gemini model consistently generates its output in a specific format, such as structured JSON from APIs or the long-form and bullet-point summaries for investment advice.
* **Data Management:**
    * **Vector Database:** `ChromaDB` (for storing text chunks and semantic embeddings for the RAG pipeline).
    * **Data Sources:** `yfinance` library, `Finnhub API`, `SerpAPI`.

## Future Plans

To further enhance performance and provide more precise financial analysis, the future roadmap includes integrating language models that have been specifically developed and fine-tuned for the finance domain. This will improve the model's understanding of financial terminology and market dynamics.

## Academic Context

This project was developed as a term project for the **SENG 472 - LLM Powered Software** course at **TED University**.

## Setup

1. Clone the repository
2. Create a virtual environment: `python -m venv venv`
3. Activate the virtual environment:
   - Windows: `venv\Scripts\activate`
   - Mac/Linux: `source venv/bin/activate`
4. Install requirements: `pip install -r requirements.txt`
5. Create a `.env` file and add your GEMINI_API_KEY, GOOGLE_API_KEY, SERP_API_KEY, FINNHUB_API_KEY.(GEMINI_API_KEY and GOOGLE_API_KEY could be same)
6. Run the chatbot via Gradio Interface: `python app.py`


## Demo Video

[Watch the project demo video here.](https://www.youtube.com/watch?v=376wNZppZ54)

