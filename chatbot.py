from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from dotenv import load_dotenv
from transformers import pipeline
import os
load_dotenv()

pipe = pipeline(
    "text-generation",
    model="mistralai/Mistral-7B-Instruct-v0.3",
    device_map="auto",
    token=os.getenv("HUGGINGFACEHUB_API_TOKEN")
)

llm = HuggingFacePipeline(pipeline=pipe)
model = ChatHuggingFace(llm=llm)


while True:
    user_input = input("You : ")
    if user_input  == "exit":
        break
    result = model.invoke(user_input)
    print("AI : ",result.content)

