import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
    api_key=os.environ.get("GROQ_API_KEY")

)

# Chain 0: Extract inventory parameters from natural language
extract_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a parameter extraction assistant. Extract inventory parameters from the user's natural language query.

Return ONLY a JSON object with these four keys (no other text):
{{"current_stock": <number>, "lead_time": <number>, "avg_daily_demand": <number>, "service_level": <number>}}

- current_stock: units currently on hand
- lead_time: days to receive a new shipment
- avg_daily_demand: average units sold/consumed per day
- service_level: target service level as a percentage (e.g. 95 for 95%)

If any parameter is missing or unclear, set its value to null."""),
    ("human", "{user_message}")
])

extract_chain = extract_prompt | llm | StrOutputParser()

# Chain 1: Compute desired stock level
compute_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an inventory planning expert. Given inventory parameters, compute the desired stock level.

Use this formula:
- Safety Stock = Z-score(service_level) * sqrt(lead_time) * average_daily_demand
- Desired Stock = (average_daily_demand * lead_time) + Safety Stock

Common Z-scores: 90% -> 1.28, 95% -> 1.65, 99% -> 2.33

Show your step-by-step calculation. End your response with a line:
DESIRED_STOCK: <number>"""),
    ("human", """Current Stock: {current_stock}
Lead Time (days): {lead_time}
Average Daily Demand: {avg_daily_demand}
Service Level: {service_level}%""")
])

compute_chain = compute_prompt | llm | StrOutputParser()

# Chain 2: Determine reorder quantity and provide recommendation
reorder_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an inventory planning advisor. Given the computation below and the current stock,
determine the reorder quantity and provide a clear recommendation.

Reorder Quantity = max(0, Desired Stock - Current Stock)

Provide:
1. The reorder quantity
2. Whether an immediate reorder is needed
3. A brief justification"""),
    ("human", """Current Stock: {current_stock}

Stock Computation:
{computation}""")
])

reorder_chain = reorder_prompt | llm | StrOutputParser()