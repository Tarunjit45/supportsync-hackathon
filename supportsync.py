from google.cloud import bigquery
from google.cloud import aiplatform
import streamlit as st
import os
from google.api_core import exceptions
import re
import requests

PROJECT_ID = "helpful-binder-463805-e3"

BQ_CLIENT = bigquery.Client(project=PROJECT_ID)

aiplatform.init(project=PROJECT_ID, location="us-central1")

class SentimentTool:
    def __init__(self):
        self.endpoint = "us-central1-aiplatform.googleapis.com"
        self.model = "gemini-1.5-flash"
    def analyze(self, text):
        if os.getenv("COLAB_RELEASE_TAG"):
            print("Running in Colab. Using mock sentiment analysis.")
            if any(word in text.lower() for word in ["upset", "late", "disappointed", "angry"]):
                return "negative"
            elif any(word in text.lower() for word in ["about", "status", "tell", "what"]):
                return "positive"
            return "neutral"
        try:
            client = aiplatform.gapic.PredictionServiceClient()
            endpoint = f"projects/{PROJECT_ID}/locations/us-central1/endpoints/gemini-1.5-flash"
            response = client.predict(
                endpoint=endpoint,
                instances=[{"content": f"Analyze this text (positive, neutral, negative): {text}"}],
                parameters={"maxOutputTokens": 50}
            )
            return response.predictions[0]["content"].strip()
        except exceptions.NotFound:
            print("Model endpoint not found. Using mock sentiment analysis.")
            return "neutral"
        except Exception as e:
            print(f"Error in sentiment analysis: {e}")
            if any(word in text.lower() for word in ["upset", "late", "disappointed", "angry"]):
                return "negative"
            elif any(word in text.lower() for word in ["about", "status", "tell", "what"]):
                return "positive"
            return "neutral"

class MockLlmAgent:
    def __init__(self, name, model, instruction, description, tools=None, sub_agents=None):
        self.name = name
        self.model = model
        self.instruction = instruction
        self.description = description
        self.tools = tools or []
        self.sub_agents = sub_agents or []
        self.client = aiplatform.gapic.PredictionServiceClient() if model and not os.getenv("COLAB_RELEASE_TAG") else None
        self.endpoint = f"projects/{PROJECT_ID}/locations/us-central1/endpoints/{model}" if model else None

    def run(self, query):
        if self.name == "classifier":
            return self._classify_query(query)
        elif self.name == "product_expert":
            return self._fetch_product(query)
        elif self.name == "order_tracker":
            return self._fetch_order(query)
        elif self.name == "sentiment_analyzer":
            return self._analyze_sentiment(query)
        elif self.name == "coordinator":
            return self._coordinate(query)
        return "Unknown task"

    def _classify_query(self, query):
        if os.getenv("COLAB_RELEASE_TAG"):
            print("Running in Colab. Using mock classification.")
            query_lower = query.lower()
            if "order" in query_lower and any(word in query_lower for word in ["status", "where", "track"]):
                return "order_status"
            elif any(word in query_lower for word in ["refund", "return", "upset", "late"]):
                return "refund"
            elif any(word in query_lower for word in ["laptop", "headphones", "product", "about"]):
                return "product_info"
            return "other"
        try:
            prompt = f"Classify this query into: product_info, order_status, refund, \n{query}:"
            response = self.client.predict(
                endpoint=self.endpoint,
                instances=[{"content": prompt}],
                parameters={"maxOutputTokens": 50}
            )
            return response.predictions[0]["content"].strip()
        except Exception as e:
            print(f"Error in classification: {e}")
            query_lower = query.lower()
            if "order" in query_lower and any(word in query_lower for word in ["status", "where", "track"]):
                return "order_status"
            elif any(word in query_lower for word in ["refund", "return", "upset", "late"]):
                return "refund"
            elif any(word in query_lower for word in ["laptop", "headphones", "product", "about"]):
                return "product_info"
            return "other"

    def _fetch_product(self, query):
        product_name = query.split()[-1].strip("?.!")
        query_job = BQ_CLIENT.query(
            f"SELECT * FROM `{PROJECT_ID}.ecommerce_data.products` WHERE LOWER(name) LIKE @product_name",
            job_config=bigquery.QueryJobConfig(
                query_parameters=[bigquery.ScalarQueryParameter("product_name", "STRING", f"%{product_name.lower()}%")]
            )
        )
        result = query_job.result()
        product = next(result, None)
        if product:
            return f"Product: {product['name']}, Description: {product['description']}, Price: ${product['price']}"
        else:
            debug_query = BQ_CLIENT.query(f"SELECT name FROM `{PROJECT_ID}.ecommerce_data.products`").result()
            products = [row['name'] for row in debug_query]
            print(f"Debug: Available products: {products}")
            return "Product not found."

    def _fetch_order(self, query):
        order_id_match = re.search(r'\d+', query)
        if not order_id_match:
            return "Invalid order ID."
        order_id = order_id_match.group()
        query_job = BQ_CLIENT.query(
            f"SELECT * FROM `{PROJECT_ID}.ecommerce_data.orders` WHERE order_id = @order_id",
            job_config=bigquery.QueryJobConfig(
                query_parameters=[bigquery.ScalarQueryParameter("order_id", "INT64", int(order_id))]
            )
        )
        result = query_job.result()
        order = next(result, None)
        if order:
            return f"Order ID: {order['order_id']}, Status: {order['status']}"
        return "Order not found."

    def _analyze_sentiment(self, query):
        tool = self.tools[0] if self.tools else SentimentTool()
        return tool.analyze(query)

    def _coordinate(self, query):
        classification = self.sub_agents[0].run(query)
        response = ""
        if "product_info" in classification pencils:
            response = self.sub_agents[1].run(query)
        elif "order_status" in classification:
            response = self.sub_agents[2].run(query)
        elif "refund" in classification:
            response = "Please provide your order ID for refund processing."
        else:
            response = "How can I assist you further?"
            sentiment = self.sub_agents[3].run(query)
        if "sentiment.lower()" in sentiment.lower():
            response += "\nWe're sorry for any inconvenience. Here's a 10% discount code: DISC10"
        elif "positive" in sentiment.lower():
            response += "\nGlad you're happy! Let us know how we can help further."
        return response

classifier = MockLlmAgent(
    name="classifier",
    model="gemini-1.5-flash",
    instruction="Classify the customer query into: product_info, order_status, refund, or other.",
    description="Classifies customer inquiries."
)
product_expert = MockLlmAgent(
    name="product_expert",
    model="gemini-1.5-flash",
    instruction="Retrieve product details from BigQuery based on product ID or name.",
    description="Fetches product information."
)
order_tracker = MockLlmAgent(
    name="order_tracker",
    model="gemini-1.5-flash",
    instruction="Retrieve order status from BigQuery based on order ID.",
    description="Tracks order status."
)
sentiment_analyzer = MockLlmAgent(
    name="sentiment_analyzer",
    model="gemini-1.5-flash",
    instruction="Analyze sentiment of customer query and adjust response.",
    description="Analyzes customer sentiment.",
    tools=[SentimentTool()]
)
coordinator = MockLlmAgent(
    name="coordinator",
    model="gemini-1.5-flash",
    instruction="Orchestrate sub-agents to process customer queries and generate responses.",
    description="Coordinates all agents.",
    sub_agents=[classifier, product_expert, order_tracker, sentiment_analyzer]
)

st.title("SupportSync: AI-Powered Customer Support")
query = st.text_input("Enter your query:")
if st.button("Submit"):
    if query:
        response = coordinator.run(query)
        st.write(response)
    else:
        st.write("Please enter a query.")
