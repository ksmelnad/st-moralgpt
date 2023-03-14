import streamlit as st
import openai
import pandas as pd
import numpy as np
from openai.embeddings_utils import get_embedding

openai.api_key = st.secrets["api_secret"]

df_embeddings = pd.read_hdf("data/df_embeddings.h5")

st.title("MoralGPT")
st.write("This app generates answers based on the Bhagavadgita commenataries using OpenAI's Gpt 3.5 turbo model.")

user_query = st.text_input("How can I help you?", )

df_similarities = df_embeddings.copy()

def get_embedding(user_query, model="text-embedding-ada-002"):
   return openai.Embedding.create(input = [user_query], model=model)['data'][0]['embedding']

def cosine_similarity(a, b):
    
    # Convert input arrays to NumPy arrays
    a = np.array(a)
    b = np.array(b)
    
    # Convert input arrays to float data type
    a = a.astype(float)
    b = b.astype(float)
    
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def search_docs(user_query, top_n):
    query_embedding = get_embedding(user_query)

    df_similarities["similarities"] = df_embeddings.ada_v2_embedding.apply(lambda x: cosine_similarity(x, query_embedding))

    res = (
        df_similarities.sort_values("similarities", ascending=False)
        .head(top_n)
    )
    
    return res

def toggle_debug():
   st.session_state['debug'] = not st.session_state['debug']

def gpt_completion():

  res = search_docs(user_query, top_n=3)
  context= res.text.values
  st.session_state['context'] = str(context)

  completion_model='gpt-3.5-turbo'

  messages=[
          {"role": "system", "content": "Answer the question as truthfully as possible using the provided context with, and if the answer is not contained within the text below, say \"I don't know."},
          {"role": "user", "content": user_query},
          {"role": "assistant", "content": "Context: \n " + st.session_state.context},
      ]

  response = openai.ChatCompletion.create(model=completion_model, temperature= 0.0, messages=messages, max_tokens=200)

  st.session_state.response = response.choices[0].message.content
  st.session_state.usage = response.usage
  return

if 'response' not in st.session_state:
  st.session_state.response = ""

if 'context' not in st.session_state:
    st.session_state['context'] = ""

if 'usage' not in st.session_state:
   st.session_state['usage'] = ""
  
if 'debug' not in st.session_state:
   st.session_state['debug'] = False
  

st.button("Submit", on_click=gpt_completion)


if st.session_state.response:
   st.markdown("**Response**")
   st.write(st.session_state.response)

if st.session_state.response:
  st.button('Toggle Debug', on_click=toggle_debug)

if st.session_state['debug']:
    st.markdown("**Usage**")
    st.write(st.session_state.usage)
    st.markdown("**Context**")
    st.write(st.session_state.context)

