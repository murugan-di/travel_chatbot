from flask import Flask, render_template, request
import openai
import pandas as pd  # For handling the dataset
import panel as pn  # GUI

# Initialize the Flask app
app = Flask(__name__)

# Set your OpenAI API key
openai.api_key = " "  # Replace with your actual OpenAI API key

# Load the dataset (ensure the dataset contains relevant information about Maldives trips)
#df = pd.read_csv("/workspaces/travel_chatbot/Maldives_Hotels_Resorts.csv")  # Replace with the actual path to your dataset

# Load the dataset (ensure the dataset contains relevant information about Maldives trips)
import chardet

# Detect encoding
with open("/workspaces/travel_chatbot/Maldives_Hotels_Resorts.csv", "rb") as file:
    result = chardet.detect(file.read())
    print(result)



# Example dataset structure:
# Name | Remarks | Cost | From Date | To Date | Adult | Child | # of nights

# Function to predict the average cost based on user inputs
def predict_average_price(df, num_adults, num_children, nights):
    # Filter the data for trips with matching conditions
    filtered_df = df[df['# of nights'] >= nights]
    filtered_df['Total Customers'] = filtered_df['Adult'] + filtered_df['Child']
    
    # Calculate the cost per customer for filtered trips
    filtered_df['Cost per Customer'] = filtered_df['Cost'] / filtered_df['Total Customers']
    
    # Predict the cost per customer and scale it based on input
    if not filtered_df.empty:
        avg_cost_per_customer = filtered_df['Cost per Customer'].mean()
        total_cost = avg_cost_per_customer * (num_adults + num_children)
        return round(total_cost, 2)
    else:
        return "No matching trips found in the dataset."

# Initialize context for TravelBot
context = [
    {'role': 'system', 'content': """
        You first greet the customer warmly, then ask for their preferences such as destination, travel dates, budget, and other requirements. \
        You provide personalized recommendations based on their inputs. \
        Clarify options to ensure all preferences are captured, such as whether flights are included, type of accommodation, and the purpose of the trip (e.g., family vacation, honeymoon, anniversary, or solo travel). \
        You summarize the travel plan and check for any additional preferences or changes before finalizing. \
        When discussing travel, make sure to cover important details such as:
        - Destinations Available are: Maldives
        - You are now capable of predicting average prices for trips based on user inputs (e.g., number of adults, children, and nights). \
        - Best seasons or months to visit
        - Budget options (with and without flights)
        - Visa requirements for the chosen destination (visa is required for each destination)
        - How far in advance to book (typically 3 months in advance)
        - Recommended number of nights and cities to visit at each destination
        - Accommodation options (e.g., best hotels for couples, family-friendly hotels)
        - Based on this input, you provide a predicted cost using your dataset. \
        - Cancellation and payment policies (e.g., single or multiple installments) (multiple installments will be available with an extra 3 percent interest rate)
        You respond in a short, friendly, and conversational style while ensuring all details are clear. \
        Your goal is to make the travel planning process smooth and enjoyable for the customer.
        Your goal is also to ask one question at a time and keep the conversation short and crisp.
    """}
]

# Function to generate OpenAI completion
def get_completion_from_messages(context, model="gpt-4o-mini"):
    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=context,
            temperature=0.2  # Degree of randomness
        )
        return response.choices[0].message["content"]
    except Exception as e:
        return f"Error: {str(e)}"

# Define the home route
@app.route("/")
def home():
    return render_template("index.html")  # Ensure you have an "index.html" template

# Define the bot response route
@app.route("/get")
def get_bot_response():
    global context
    userText = request.args.get("msg")  # Get user input from the query string
    if userText:
        # Add user input to the context
        context.append({'role': 'user', 'content': userText})
        
        # Extract preferences if mentioned
        if "adults" in userText.lower() and "children" in userText.lower() and "nights" in userText.lower():
            try:
                # Extract numbers for adults, children, and nights (you can improve this with regex or NLP techniques)
                num_adults = int(userText.split("adults")[0].strip().split()[-1])
                num_children = int(userText.split("children")[0].strip().split()[-1])
                nights = int(userText.split("nights")[0].strip().split()[-1])
                
                # Predict the average price
                predicted_price = predict_average_price(df, num_adults, num_children, nights)
                
                # Generate a response with the predicted price
                response = f"Based on your inputs, the estimated total cost for your trip to the Maldives is ₹{predicted_price}."
            except Exception as e:
                response = "Sorry, I couldn't understand your input. Could you provide the number of adults, children, and nights clearly?"
        else:
            # Default OpenAI completion for general queries
            response = get_completion_from_messages(context)
        
        # Add bot response to the context
        context.append({'role': 'assistant', 'content': response})
        
        return response
    return "Error: No input provided."

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True)  # Use debug=True for easier debugging during development
