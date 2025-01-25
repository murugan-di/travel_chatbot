from flask import Flask, render_template, request
import os
import openai
import pandas as pd
import pdfplumber
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_community.chat_models import ChatOpenAI
from langchain.text_splitter import CharacterTextSplitter
import chardet  # For encoding detection
from langchain.vectorstores import FAISS
import logging

# Initialize the Flask app
app = Flask(__name__)

# Set your OpenAI API key
openai.api_key = " "

# Define the conversational prompt
prompt_context = [
    {'role': 'system', 'content': """
        You are TravelBot, an automated service to assist users with planning their travel. \
        You first greet the customer warmly, then ask for their preferences such as destination, travel dates, budget, and other requirements. \
        You provide personalized recommendations based on their inputs. \
        Clarify options to ensure all preferences are captured, such as whether flights are included, type of accommodation, and the purpose of the trip (e.g., family vacation, honeymoon, anniversary, or solo travel). \
        You summarize the travel plan and check for any additional preferences or changes before finalizing. \
        When discussing travel, make sure to cover important details such as:
        - Destinations Available are: Dubai, Bali, Thailand, Maldives
        - Budget for each of them are 50000, 65000, 80000, 100000 rupees respectively (with flights).
        - For flights, it is 10000 rupees extra per person.  
        - Best seasons or months to visit
        - Budget options (with and without flights)
        - Visa requirements for the chosen destination (visa is required for each destination)
        - How far in advance to book (typically 3 months in advance)
        - Recommended number of nights and cities to visit at each destination
        - Accommodation options (e.g., best hotels for couples, family-friendly hotels)
        - Special occasions (e.g., honeymoon or anniversary packages)
        - For special occasions, there is a discount of 10%. 
        - Cancellation and payment policies (e.g., single or multiple installments) (multiple installments will be available with an extra 3 percent interest rate)
        You respond in a short, friendly, and conversational style while ensuring all details are clear. \
        Your goal is to make the travel planning process smooth and enjoyable for the customer.
        Your goal is also to ask one question at a time and keep the conversation short and crisp.
    """}
]


def load_csv_data(file_path):
    # Detect file encoding
    with open(file_path, "rb") as file:
        result = chardet.detect(file.read())
        encoding = result["encoding"]
        if encoding is None:  # Fallback if detection fails
            encoding = "utf-8"  # Or another encoding that fits your data
        print(f"Detected encoding: {encoding}")

    # Read the CSV using the detected or fallback encoding
    df = pd.read_csv(file_path, encoding=encoding)

    # Convert each row into a concatenated string
    text_data = []
    for index, row in df.iterrows():
        text_data.append(" | ".join([str(value) for value in row]))
    return text_data

def load_pdf_data(file_path):
    try:
        text_data = ""
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                text_data += page.extract_text()
        if not text_data.strip():
            print(f"Warning: {file_path} is empty or contains no readable text.")
        return text_data
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return ""  # Return empty string for invalid PDFs


# Load and preprocess data
csv_file_path = "/workspaces/travel_chatbot/Maldives_Hotels_Resorts.csv"
pdf_file_path1 = "/workspaces/travel_chatbot/Maldives_Itenary.pdf"
pdf_file_path2 = "/workspaces/travel_chatbot/Cocogiri-Island-Resort.pdf"
pdf_file_path3 = "/workspaces/travel_chatbot/Sun_sivam.pdf"
pdf_file_path4 = "/workspaces/travel_chatbot/reethi_faru_resort.pdf"

# Define all PDF file paths
pdf_file_paths = [
    "/workspaces/travel_chatbot/Maldives_Itenary.pdf",
    "/workspaces/travel_chatbot/Cocogiri-Island-Resort.pdf",
    "/workspaces/travel_chatbot/Sun_sivam.pdf",
    "/workspaces/travel_chatbot/reethi_faru_resort.pdf"
]

# Load all PDFs dynamically
pdf_data = [load_pdf_data(path) for path in pdf_file_paths]

# Combine all loaded PDF data
combined_data = pdf_data


# Split the data into smaller chunks for embedding
text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=100, length_function=len)
documents = text_splitter.split_text("\n".join(combined_data))

# Create embeddings and vectorstore
embeddings = OpenAIEmbeddings(openai_api_key=openai.api_key)
# Create FAISS vectorstore
vectorstore = FAISS.from_texts(documents, embeddings)

# Create retriever and QA chain
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.3, openai_api_key=openai.api_key)
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)


# Function to get OpenAI chat completion
def get_openai_completion(context):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=context,
            temperature=0.4
        )
        return response.choices[0].message["content"]
    except Exception as e:
        return f"Error: {str(e)}"
    

# Function to log chat messages
def log_chat(user_message, bot_response):
    with open("chat_log.txt", "a") as log_file:
        log_file.write(f"User: {user_message}\n")
        log_file.write(f"Bot: {bot_response}\n")
        log_file.write("=" * 50 + "\n")  # Separator for readability

# Define the home route
@app.route("/")
def home():
    return render_template("index.html")  # Ensure you have an "index.html" template

# Define the bot response route
@app.route("/get")
def get_bot_response():
    userText = request.args.get("msg")  # Get user input from the query string
    if userText:
        try:
            # Add user input to prompt context
            prompt_context.append({'role': 'user', 'content': userText})

            # Use retrieval chain for document-based queries
            retrieved_response = qa_chain.run(userText)

            # If no relevant document is retrieved, use OpenAI prompt context
            if retrieved_response.strip() == "":
                print("No relevant documents retrieved. Using fallback prompt.")
                response = get_openai_completion(prompt_context)
            else:
                response = retrieved_response

            # Log the conversation
            log_chat(userText, response)

            # Add bot response to the context
            prompt_context.append({'role': 'assistant', 'content': response})

            return response
        except Exception as e:
            return f"Error: {str(e)}"
    return "Error: No input provided."

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True)  # Use debug=True for easier debugging during development
