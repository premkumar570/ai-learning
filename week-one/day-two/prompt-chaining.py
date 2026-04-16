import os 
from dotenv import load_dotenv
from openai import OpenAI
from landchain_qroq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough


load_dotenv()

llm =ChatGroq(
    model="openai/gpt-oss-120b",
    api_key=os.environ.get("GROQ_API_KEY")
)

summary_chain=ChatPromptTemplate(
    input_variables=["review"],
    template="Summarize the following cumtomer review into a single    sentence:\n\n{review} \n\nsummery"
) | llm
sentiment_chain=ChatPromptTemplate(
    input_variables=["sumery"],
    template="classify the sentiment of the following review as positive, negative, or neutral:\n\n{sumery} \n\nsentiment"
) | llm

suggestion_chain=ChatPromptTemplate(
    input_variables=["sentiment"],
    template="based on the sentiment, provide a suggestion for the following review:\n\n{sentiment} \n\nsuggestion"
) | llm


full_chain = summary_chain 
| RunnablePassthrough(lambda x: {"sumery": x}) 
| RunnablePassthrough(lambda x: {"sentiment": x}) 
| sentiment_chain 
| RunnablePassthrough(lambda x: {"suggestion": x}) 
| suggestion_chain

review = "The product was okay, but the delivery was slow."

result = full_chain.invoke({"review": review})

print(result)


# client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# load_dotenv()