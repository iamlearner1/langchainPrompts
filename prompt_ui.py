from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
import streamlit as st
from langchain_core.prompts import load_prompt
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

st.header('Reasearch Tool')

paper_input = st.selectbox( "Select Research Paper Name", ["Attention Is All You Need", "BERT: Pre-training of Deep Bidirectional Transformers", "GPT-3: Language Models are Few-Shot Learners", "Diffusion Models Beat GANs on Image Synthesis"] )

style_input = st.selectbox( "Select Explanation Style", ["Beginner-Friendly", "Technical", "Code-Oriented", "Mathematical"] ) 

length_input = st.selectbox( "Select Explanation Length", ["Short (1-2 paragraphs)", "Medium (3-5 paragraphs)", "Long (detailed explanation)"] )


template =  load_prompt('template.json')

prompt = template.invoke({

    'paper_input' : paper_input,
    'style_input' : style_input,
    'length_input' : length_input
})



if st.button("summarize"):
    result = model.invoke(prompt)
    st.write(result.content)