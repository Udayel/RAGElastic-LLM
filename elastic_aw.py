from langchain.chains import RetrievalQA
from langchain.chains import LLMChain
from langchain import OpenAI
from langchain.prompts import PromptTemplate
from langchain import SagemakerEndpoint
from langchain.llms.sagemaker_endpoint import LLMContentHandler
import json
import os
import streamlit as st
from elasticsearch import Elasticsearch
from typing import List, Tuple, Dict
import boto3
import botocore


flan_t5_endpoint_name = os.environ["FLAN_T5_ENDPOINT"]
falcon_40b_endpoint_name = os.environ["FALCON_40B_ENDPOINT"]
aws_region = os.environ["AWS_REGION"]
cid = os.environ['ES_CLOUD_ID']
cp = os.environ['ES_PASSWORD']
cu = os.environ['ES_USERNAME']
max_tokens=1024
max_context_tokens=2000
safety_margin=5

LLM_LIST: List[str] = ["Falcon-40B-Instruct","Flan-T5-XXL"]
INDEX_LIST: List[str] = ["search-elastic-docs", "anyitcompany"]

class ContentHandlerFalcon(LLMContentHandler):
        content_type = "application/json"
        accepts = "application/json"

        def transform_input(self, prompt: str, model_kwargs: dict) -> bytes:
            input_str = json.dumps({"inputs": prompt, **model_kwargs})
            return input_str.encode('utf-8')

        def transform_output(self, output: bytes) -> str:
            response_json = json.loads(output.read().decode("utf-8"))
            return response_json[0]["generated_text"]

class ContentHandlerFlan(LLMContentHandler):
        content_type = "application/json"
        accepts = "application/json"

        def transform_input(self, prompt: str, model_kwargs: dict) -> bytes:
            input_str = json.dumps({"text_inputs": prompt, **model_kwargs})
            return input_str.encode('utf-8')

        def transform_output(self, output: bytes) -> str:
            response_json = json.loads(output.read().decode("utf-8"))
            return response_json["generated_texts"][0]


# Connect to Elastic Cloud cluster
def es_connect(cid, user, passwd):
    es = Elasticsearch(cloud_id=cid, http_auth=(user, passwd))
    print("Connection is", es)
    return es

# Search ElasticSearch index and return body and URL of the result
def search(query_text):


    print("Query text is", query_text)
    es = es_connect(cid, cu, cp)



    # Elasticsearch query (BM25) and kNN configuration for hybrid search
    query = {
        "bool": {
            "must": [{
                "match": {
                    "title": {
                        "query": query_text,
                        "boost": 1
                    }
                }
            }]
        }
    }

    knn = {
        "field": "title-vector",
        "k": 1,
        "num_candidates": 20,
        "query_vector_builder": {
            "text_embedding": {
                "model_id": "1",
                "model_text": query_text
            }
        },
        "boost": 24
    }

    fields = ["title", "body_content", "url"]
    index = elastic_index
    resp = es.search(index=index,
                     query=query,
                     knn=knn,
                     fields=fields,
                     size=1,
                     source=False)

    print("Query is", query)
    print("Response is",resp)
    body = resp['hits']['hits'][0]['fields']['body_content'][0]
    url = resp['hits']['hits'][0]['fields']['url'][0]

    print("Body is",body)
    print("URL is",url)

    return body, url

def truncate_text(text, max_tokens):
    tokens = text.split()
    print('Number of tokens',len(tokens))
    if len(tokens) < max_tokens:
        return text

    print('Number of tokens after truncation',len(tokens[:512]))

    return ' '.join(tokens[:512])



st.title("Anycompany AI Assistant")

with st.sidebar.expander("⚙️", expanded=True):
    llm_model = st.selectbox(label='Choose Large Language Model', options=LLM_LIST)
    elastic_index = st.selectbox(label='Choose ElasticSearch Index', options=INDEX_LIST)
    with_context = st.checkbox('With Context')


print("Selected LLM Model is:",llm_model)
print("Selected Elastic Index is:",elastic_index)
print("Selected Context Option is:",with_context)

if llm_model == "Flan-T5-XXL":
    endpoint_name = flan_t5_endpoint_name
    content_handler = ContentHandlerFlan()
elif llm_model == "Falcon-40B-Instruct":
    endpoint_name = falcon_40b_endpoint_name
    content_handler = ContentHandlerFalcon()
else:
    endpoint_name = "NA"

# Main chat form
with st.form("chat_form"):
    query = st.text_input("You: ")
    submit_button = st.form_submit_button("Send")

# Generate and display response on form submission
negResponse = "I'm unable to answer the question based on the information I have from Elastic Docs."
if submit_button:
    resp, url = search(query)
    print("Response before truncation:",resp)
    resp = truncate_text(resp, max_context_tokens - max_tokens - safety_margin)
    print("Response after truncation:",resp)
    prompt_without_context = f"Answer this question: {query} \n"
    prompt_with_context = f"Answer this question: {query}\nUsing only the information from this Elastic Doc: {resp}\n"

    prompt = prompt_without_context

    if with_context:
        prompt = prompt_with_context
    #answer = chat_gpt(prompt)

    ####adding here
    #content_handler = ContentHandler()


    llm=SagemakerEndpoint(
            endpoint_name=endpoint_name,
            #region_name="us-east-1",
            region_name=aws_region,
            model_kwargs={"temperature":1, "max_length": 2048},
            content_handler=content_handler
        )
    answer = llm(prompt)
    print("Answer is",answer)

    ####stopping here

    if negResponse in answer:
        st.write(f"AI: {answer.strip()}")
    else:
        if with_context:
            st.write(f"AI: {answer.strip()}\n\nDocs: {url}")
        else:
            st.write(f"AI: {answer.strip()}\n")
