from flask import Flask, render_template, request
import openai
import panel as pn  # GUI

# Initialize the Flask app
app = Flask(__name__)

# Set your OpenAI API key
openai.api_key = " "


# Initialize context for TravelBot
context = [
    {'role': 'system', 'content': """
        You are TravelBot, an automated service to assist users with planning their travel. \
        You first greet the customer warmly, then ask for their preferences such as destination, travel dates, budget, and other requirements. \
        You provide personalized recommendations based on their inputs. \
        Clarify options to ensure all preferences are captured, such as whether flights are included, type of accommodation, and the purpose of the trip (e.g., family vacation, honeymoon, anniversary, or solo travel). \
        You summarize the travel plan and check for any additional preferences or changes before finalizing. \
        When discussing travel, make sure to cover important details such as:
        - Destinations Available are: Dubai, Bali, Thailand, Maldives
        - Budget for each of them are 50000, 65000, 80000, 100000 rupees respectively (with flights).
        - For with flights it is 10000 rupees extra per person.  
        - Best seasons or months to visit
        - Budget options (with and without flights)
        - Visa requirements for the chosen destination (visa is required for each destination)
        - How far in advance to book (typically 3 months in advance)
        - Recommended number of nights and cities to visit at each destination
        - Accommodation options (e.g., best hotels for couples, family-friendly hotels)
        - Special occasions (e.g., honeymoon or anniversary packages)
        - For special ocassions - there is a discount of 10%. 
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
            temperature=0  # Degree of randomness
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
    global context
    userText = request.args.get("msg")  # Get user input from the query string
    if userText:
        # Add user input to the context
        context.append({'role': 'user', 'content': userText})
        
        # Get response from OpenAI
        response = get_completion_from_messages(context)
        
        # Add bot response to the context
        context.append({'role': 'assistant', 'content': response})
        
        # Log the conversation
        log_chat(userText, response)
        
        return response
    return "Error: No input provided."

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True)  # Use debug=True for easier debugging during development


