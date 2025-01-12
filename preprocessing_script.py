# Example of preprocessing text data into chunks
def preprocess_text(file_path, chunk_size=500):
    with open(file_path, "r") as file:
        text = file.read()
    
    # Split the text into chunks of specified size
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    return chunks

# Preprocess and store chunks in Pinecone
text_chunks = preprocess_text("/workspaces/travel_chatbot/chat_log.txt")
embed_and_store(text_chunks)
