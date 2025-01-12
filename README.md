TravelBot: A Retrieval-Augmented Generation (RAG) Chatbot

TravelBot is a conversational chatbot designed to assist users with personalized travel planning. It uses a Retrieval-Augmented Generation (RAG) workflow, integrating Pinecone for vector search and OpenAI's GPT model to generate contextually relevant responses. TravelBot is implemented with Flask and can preprocess text files for embedding-based retrieval.
Features

    Conversational Travel Assistant: Provides travel recommendations based on user preferences, such as destination, budget, and dates.
    Context-Aware Responses: Uses a combination of system context and dynamically retrieved data for relevant and accurate responses.
    Vector Search with Pinecone: Stores and retrieves text embeddings for efficient context retrieval.
    OpenAI GPT Integration: Generates natural language responses using OpenAI's GPT-3.5 Turbo model.
    Log Management: Logs user queries and bot responses to chat_log.txt for reference.

Requirements

To run TravelBot, you need the following dependencies installed:

    Python 3.8 or higher
    Flask
    OpenAI Python library == 0.28
    Pinecone Python library

You can install the required libraries using pip:

pip install flask openai pinecone-client

Setup Instructions
1. Clone the Repository

Clone this repository to your local machine:

git clone https://github.com/murugan-di/travel_chatbot.git
cd travelbot

2. Configure API Keys

Update the API keys in app.py:

    OpenAI API Key: Replace your-openai-api-key with your OpenAI key.
    Pinecone API Key: Replace your-pinecone-api-key with your Pinecone key.
    Pinecone Environment: Update the Pinecone environment (e.g., us-east-1).

3. Preprocess Text Data

Ensure you have a text file (chat_log.txt) for storing travel-related information. The app will preprocess and upload this data to Pinecone during initialization.
4. Run the Application

Start the Flask application:

python app.py

The app will run locally at http://127.0.0.1:5000.
Usage

    Open your browser and navigate to the running Flask app.
    Type your travel-related queries in the chatbox.
    The chatbot will respond with personalized travel recommendations and guidance.

Logging

All user queries and chatbot responses are logged in the chat_log.txt file. Logs include:

    User messages
    Bot responses
    Separators for readability
