from flask import Flask, render_template, request
import openai
from pinecone import Pinecone, ServerlessSpec
import logging

# Flask app initialization
app = Flask(__name__)

# Set your API keys
openai.api_key = " "
pinecone = Pinecone(api_key=" ")


# Pinecone index setup
index_name = "travel-chatbot"
if index_name not in pinecone.list_indexes().names():
    logging.info(f"Index '{index_name}' not found. Creating it.")
    pinecone.create_index(
        name=index_name,
        dimension=1536,  # Dimension for OpenAI embeddings
        metric="cosine",  # Cosine similarity
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

index = pinecone.Index(index_name)

# Chatbot context configuration
system_context = """
    You are TravelBot, an automated service to assist users with planning their travel. \
    Your primary goal is to provide personalized travel recommendations based on user preferences. \
    Always respond in a friendly, concise, and conversational tone.
"""

# Function to embed and store data in Pinecone
def embed_and_store(text_chunks):
    try:
        for i, chunk in enumerate(text_chunks):
            # Generate embedding for the text chunk
            embedding = openai.Embedding.create(
                input=chunk,
                model="text-embedding-ada-002"
            )["data"][0]["embedding"]
            logging.debug(f"Generated embedding for chunk {i}")

            # Store the embedding in Pinecone
            index.upsert([(f"chunk-{i}", embedding, {"text": chunk})])
            logging.info(f"Stored chunk {i} in Pinecone.")
    except Exception as e:
        logging.error(f"Error in embed_and_store: {str(e)}")

# Function to preprocess text into manageable chunks
def preprocess_text(file_path, chunk_size=500):
    try:
        with open(file_path, "r") as file:
            text = file.read()
        # Split the text into chunks of specified size
        chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
        logging.info(f"Preprocessed text into {len(chunks)} chunks.")
        return chunks
    except Exception as e:
        logging.error(f"Error in preprocess_text: {str(e)}")
        return []

# Preprocess and store chunks in Pinecone
text_chunks = preprocess_text("chat_log.txt")  # Replace with your text file path
embed_and_store(text_chunks)

# Function to retrieve context from Pinecone
def retrieve_context(user_query):
    try:
        # Generate embedding for the user's query
        query_embedding = openai.Embedding.create(
            input=user_query,
            model="text-embedding-ada-002"
        )["data"][0]["embedding"]
        logging.debug(f"Generated embedding for user query.")

        # Query Pinecone for the most relevant chunks
        results = index.query(query_embedding, top_k=3, include_metadata=True)
        logging.debug(f"Pinecone query results: {results}")

        # Combine the retrieved text into a single context
        context = " ".join([match["metadata"]["text"] for match in results["matches"]])
        logging.info(f"Retrieved context for query: {user_query}")
        return context
    except Exception as e:
        logging.error(f"Error in retrieve_context: {str(e)}")
        return ""

# Function to generate a response from the LLM
def generate_response(user_query):
    try:
        # Retrieve relevant context for the query
        context = retrieve_context(user_query)
        logging.debug(f"Retrieved context: {context}")

        # Prepare the prompt with system context, retrieved context, and user query
        prompt = f"""
        System context: {system_context}
        Retrieved context: {context}
        User query: {user_query}
        Please provide a concise and helpful response to the user query.
        """
        logging.debug(f"Generated prompt for LLM: {prompt}")

        # Query the OpenAI model
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )
        bot_response = response.choices[0].message["content"]
        logging.info(f"Generated response: {bot_response}")
        return bot_response
    except Exception as e:
        logging.error(f"Error in generate_response: {str(e)}")
        return f"Error: {str(e)}"

# Function to log chat messages to a file
def log_chat(user_message, bot_response):
    try:
        with open("chat_log.txt", "a") as log_file:
            log_file.write(f"User: {user_message}\n")
            log_file.write(f"Bot: {bot_response}\n")
            log_file.write("=" * 50 + "\n")  # Separator for readability
        logging.info("Logged chat to chat_log.txt.")
    except Exception as e:
        logging.error(f"Error in log_chat: {str(e)}")

# Flask routes
@app.route("/")
def home():
    return render_template("index.html")  # Ensure you have a corresponding `index.html`

@app.route("/get")
def chatbot_response():
    user_query = request.args.get("msg")
    if user_query:
        logging.debug(f"Received user query: {user_query}")
        try:
            # Generate a response using the RAG workflow
            bot_response = generate_response(user_query)
            logging.debug(f"Generated bot response: {bot_response}")

            # Log the chat
            log_chat(user_query, bot_response)

            return bot_response
        except Exception as e:
            logging.error(f"Error generating response: {str(e)}")
            return f"Error: {str(e)}"
    logging.warning("No query provided by the user.")
    return "Error: No query provided."

# Flask app runner
if __name__ == "__main__":
    app.run(debug=True)
