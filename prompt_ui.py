from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
import streamlit as st
llm = HuggingFacePipeline.from_model_id(
    model_id='TinyLlama/TinyLlama-1.1B-Chat-v1.0',
    task='text-generation',
    # pipeline_kwargs = dict(
    #     temperature = 0.5,
    #     max_new_tokens = 300
    # )
)

model = ChatHuggingFace(llm=llm)
# result = model.invoke("what do you mean by affirmation")
# print(result)

st.header("Text Summarizer")
user_input = st.text_input("Enter your prompt ")

if st.button("summarize"):
    result = model.invoke(user_input)
    st.write(result.content)