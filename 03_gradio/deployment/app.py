import evaluate
import time
import numpy as np
import torch
import re
import os
import json
import boto3
import openai
from openai import OpenAI

import nltk
from nltk.tokenize import sent_tokenize
nltk.download("punkt")

from transformers import (
    AutoModelForSeq2SeqLM
    ,AutoTokenizer
    ,DataCollatorForSeq2Seq
    ,Seq2SeqTrainer
    ,Seq2SeqTrainingArguments
    ,BitsAndBytesConfig
    ,AutoModelForCausalLM
    ,GenerationConfig
)

from peft import (
    PeftConfig
    ,PeftModel
    ,LoraConfig
    ,get_peft_model
)

from transformers import pipeline
import gradio as gr

from datetime import datetime
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi

import chromadb
from chromadb.utils import embedding_functions

from css_html import custom_css

#-----------------#
#--- RAG SETUP ---#
#-----------------#

def download_s3_folder(bucket_name, s3_folder, local_dir):
    """
    Download the contents of a folder directory from S3 to a local directory.

    Parameters:
        bucket_name (str): Name of the S3 bucket.
        s3_folder (str): Path of the folder in the S3 bucket.
        local_dir (str): Local path to which the files will be downloaded.
    """
    # Initialize a session using your credentials
    session = boto3.Session(
        aws_access_key_id = os.environ['aws_access_key_id'],
        aws_secret_access_key = os.environ['aws_secret_access_key']
    )

    # Create an S3 client
    s3 = session.client('s3')

    # Ensure the local directory exists
    if not os.path.exists(local_dir):
        os.makedirs(local_dir)

    # Handle the trailing slash
    if not s3_folder.endswith('/'):
        s3_folder += '/'

    # List objects within a given prefix
    paginator = s3.get_paginator('list_objects_v2')
    for page in paginator.paginate(Bucket=bucket_name, Prefix=s3_folder):
        for obj in page.get('Contents', []):
            key = obj['Key']
            if not key.endswith('/'):  # skip directories
                # Construct the full path to save the file
                local_file_path = os.path.join(local_dir, key[len(s3_folder):])
                local_file_dir = os.path.dirname(local_file_path)

                # Ensure the local directory exists
                if not os.path.exists(local_file_dir):
                    os.makedirs(local_file_dir)

                # Download the file
                s3.download_file(bucket_name, key, local_file_path)
                # print(f"Downloaded {key} to {local_file_path}")

# load RAG db
bucket_name = 'projectmarley'
s3_folder = 'data/chroma_db/'
local_dir = './chroma_db'

download_s3_folder(bucket_name, s3_folder, local_dir)

# embed data into RAG vector store
sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="sentence-transformers/all-MiniLM-L6-v2")

chroma_client = chromadb.PersistentClient(path="./chroma_db")
chroma_collection = chroma_client.get_or_create_collection(
    name="quickstart",
    embedding_function=sentence_transformer_ef)

# define function to retrieve documents from RAG vector db
def retrieve_doc(query):
    '''
    Purpose: to retrieve most relevant and similar document from db vector store
    @params query: the query to search on (str)
    returns: a document (str)
    '''
    result = chroma_collection.query(query_texts=[query], n_results=1)

    return result['metadatas'][0][0]['answer']


#------------------------#
#--- GRADIO APP SETUP ---#
#------------------------#

def truncate_text(text):
    '''
    Purpose: to truncate and remove incomplete sentences
    @params text: text to truncate (str)
    returns: truncated text (str)
    '''
    # Split the text into individual sentences.
    sentences = re.split('(?<=[.?!])\s+', text)

    # Container for the truncated response.
    keep_sentences = []
    for sentence in sentences:
        if sentence.endswith('.'):  # determine if this sentence ends with a period; it is likely the end of the response section.
            keep_sentences.append(sentence)
        else:
            pass

    if len(keep_sentences) == 0:
        return sentences[0]
    else:
        return ' '.join(keep_sentences)  # join the processed sentences back into a coherent string
    

def insert_data(data, db, coll):
    '''
    Purpose: to insert user feedback data to mongodb database
    @params data: the json data to insert (dict)
    @params db: the name of the database in mongodb to insert data to (str)
    @params coll: the name of the collection in mongodb to insert data to (str)
    returns: nothing or error message
    '''
    uri = os.environ['MONGODB_URI']

    # Create a new client and connect to the server
    client = MongoClient(uri, server_api=ServerApi('1'))

    try:
        # set database & collection
        database = client[db]
        collection = database[coll]

        # insert data
        collection.insert_one(data)

        client.close()

    except Exception as e:
        return str(e)
    

def flanT5_predict(pipe, message):
    '''
    Purpose: to use fine-tuned Flan-T5 model to generate response
    @params pipe: the transformers pipe object to generate predictions
    @params message: user input/question (str)
    returns: a response/answer (str)
    '''
    doc = retrieve_doc(message)  # retrieve document from RAG db store

    ans = pipe(f'Based on this context: {doc}. Respond to this question: {message}'
               ,max_time=7           # max time for computation in seconds
               ,max_new_tokens=512   # max token length to generate
               ,no_repeat_ngram_size=2
            #    ,temperature=0.1
            #    ,do_sample=True
            #    ,early_stopping=True
            #    ,top_k=3
               )

    ans = truncate_text(ans[0]['generated_text']) # remove incomplete sentences caused by model generation limitation

    return ans


def falcon_predict(model, tokenizer, message):
    '''
    Purpose: to use fine-tuned Falcon 7B model to generate response
    @params model: the loaded PEFT model
    @params tokenizer: the PEFT tokenizer of the model
    @params message: user input/question (str)
    returns: a response/answer (str)
    '''
    doc = retrieve_doc(message)  # retrieve document from RAG db store

    prompt = f"Using the following knowledge: {doc}. Respond to the following prompt: {message}"

    input_ids = tokenizer.encode(prompt
                                 ,return_tensors="pt")

    eos_token_id = tokenizer.eos_token_id

    outputs = model.generate(
        input_ids=input_ids
        ,max_new_tokens=512
        ,max_time=8
    )

    # decode the generated response
    generated_responses = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    # extract the answer from generated text
    ans = generated_responses[0].replace(prompt,'').replace('Using the following knowledge: ','')
    ans = truncate_text(ans) # remove incomplete sentences caused by model generation limitation

    return ans


def llama2_predict(pipe, message):
    '''
    Purpose: to use fine-tuned LlaMA-2-7B model to generate response
    @params pipe: the transformers pipe object to generate predictions
    @params message: user input/question (str)
    returns: a response/answer (str)
    '''
    doc = retrieve_doc(message)  # retrieve document from RAG db store

    ans = pipe(f'<s>[INST] {message} Make an answer referring this {doc} without overlapping contents [/INST]'
                ,max_new_tokens=150)

    ans = truncate_text(ans[0]['generated_text'].split("[/INST] ")[1]) # remove incomplete sentences caused by model generation limitation

    return ans


def gpt_predict(message):
    '''
    Purpose: to use fine-tuned GPT3.5-Turbo model to generate response
    @params message: user input/question (str)
    returns: a response/answer (str)
    '''
    # connect to openai client
    client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])
    doc = retrieve_doc(message)  # retrieve document from RAG db store

    # pass RAG documents and query to GPT via openai API
    messages = [
        {
            "role": "system",
            "content": "As a skilled veterinarian specialized in dogs, give answer to the following question."
        },
        {"role": "user", "content": message},
        {"role": "system", "content": "Here are examples of answers" + doc +
                                      "Taking this into account, provide an integrated answer to the user's question."}
    ]
    response = client.chat.completions.create(model="ft:gpt-3.5-turbo-0613:personal::93dH1NCq", messages=messages, max_tokens=500)

    # Handling the response
    if 'choices' in response and response['choices']:
        first_choice = response['choices'][0]
        if 'message' in first_choice and 'content' in first_choice['message']:
            return first_choice['message']['content']
    
    ans = response.choices[0].message.content
    
    return ans


#------------------------------------#
#--- load FLAN-T5 fully-finetuned ---#
#------------------------------------#
base_model_name = "google/flan-t5-base"

flan_tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
flan_tokenizer.pad_token = flan_tokenizer.eos_token
flan_tokenizer.padding_side = "right"  # set padding to the right to avoid issues with fp16 (esp. when using 4-bit quantization)

# set up LLM pipeline with fine-tuned model
flan_model = "dahlia25/flan-t5-base-dog-full"

flan_t5_pipe = pipeline(task='text2text-generation'  # not 'question-answer' for T5 models
                        ,model=flan_model
                        ,tokenizer=flan_tokenizer)


#----------------------------#
#--- load Falcon-7B QLoRA ---#
#----------------------------#
model = "deepaknh/falcon7B_FineTuning_Experiment2_QLORA_7perParam"

config = PeftConfig.from_pretrained(model)
bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )

peft_base_model = AutoModelForCausalLM.from_pretrained(
        config.base_model_name_or_path,
        return_dict=True,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

falcon_peft_model = PeftModel.from_pretrained(peft_base_model, model)

falcon_peft_tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
falcon_peft_tokenizer.pad_token = falcon_peft_tokenizer.eos_token


#----------------------------#
#--- load LlaMA2-7B QLoRA ---#
#----------------------------#
model_name = "NousResearch/Llama-2-7b-chat-hf"
adapters_name = "Sjbok/Llama_2_7B_PEFT_QLORA_V2"

peft_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map={"": 0}
)
model = PeftModel.from_pretrained(peft_model, adapters_name)
llama_model = model.merge_and_unload()

llama_tokenizer = AutoTokenizer.from_pretrained(model_name)
llama_tokenizer.pad_token = llama_tokenizer.eos_token

llama_pipe = pipeline(task='text-generation'
                        ,model=llama_model
                        ,tokenizer=llama_tokenizer)


#------------------------#
#--- Chatbot UI Setup ---#
#------------------------#

def print_like_dislike(x: gr.LikeData):
    '''
    Purpose: to provide users to like and dislike model responses, and save like/dislike user data
    @params x: the LikeData gradio instance
    returns: nothing; print likes/dislikes to reflect in chatbot app
    '''
    data = {
        'datetime': datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
        ,"response":str(x.value)
        ,"liked":str(x.liked)
    }
    
    # insert data to mongodb
    db = 'GRADIO_APP'
    coll = 'USER_FEEDBACK'
    resp = insert_data(data, db, coll)


def calc_feedback_stats(assistant):
    '''
    Purpose: to generate a summary statistics in text
    @params assistant: the name of assistant corresponding to LLM model (str)
    returns: a summarized statistics in text (str), and % of liked responses (float)
    '''
    uri = os.environ['MONGODB_URI']

    # Create a new client and connect to the server
    client = MongoClient(uri, server_api=ServerApi('1'))

    # set database & collection
    database = client['GRADIO_APP']
    collection = database['USER_FEEDBACK']

    # get number of liked responses
    condition1 = {"response": { "$regex": f"{assistant}:" }}
    condition2 = {"liked": "True"}

    query = {"$and": [condition1, condition2]}
    result = collection.find(query)
    result = list(result)

    num_likes = len(result)

    # get number of disliked responses
    condition1 = {"response": { "$regex": f"{assistant}:" }}
    condition2 = {"liked": "False"}

    query = {"$and": [condition1, condition2]}
    result = collection.find(query)
    result = list(result)

    num_dislikes = len(result)

    # get total number of responses
    database = client['GRADIO_APP']
    collection = database['QA_GENERATED']

    condition1 = {"response": { "$regex": f"{assistant}:" }}

    query = {"$and": [condition1]}
    result = collection.find(query)
    result = list(result)

    total_responses = len(result)
    
    try:
        like_perc = num_likes/(num_likes + num_dislikes)
    except: # if assistant does not have any rated responses and cannot divide by zero
        like_perc = 0

    client.close()

    return (f'answered {total_responses} questions (with {num_likes} likes + {num_dislikes} dislikes)', round(like_perc,2))


def create_labels():
    '''
    Purpose: generate a dictionary of labels to input to gradio label component
    returns: a dictionary of labels (dict)
    '''
    assistant_list = ['Flo', 'Bubba', 'Lois', 'Biscuit']
    
    label_dict = {}
    for assistant in assistant_list:
        text_summary, rating = calc_feedback_stats(assistant)
        label_dict[f'{assistant}: {text_summary}'] = rating  # rating = the % of responses that were liked by users

    return label_dict


def greet(message, assistant):
    '''
    Purpose: to generate greeting responses
    @params message: user input/prompt (str)
    @params assistant: the selected assistant (aka. model) from model_select dropdown (str)
    returns: a greeting (str)
    '''
    # check if message is a greeting
    one_word_greetings = ['hi', 'hello', 'hey', 'greetings']
    two_word_greetings = ['good evening', 'good afternoon', 'good morning']
    
    reformat_message = message.lower().split()[:1]
    greeting_ind = any(greeting in reformat_message for greeting in one_word_greetings)

    reformat_message = ' '.join(message.lower().split()[:2])
    greeting_ind += any(greeting in reformat_message for greeting in two_word_greetings)

    if greeting_ind > 0:  # message is a greeting
        return f"Hi, I'm {assistant}. What dog questions can I help answer?"
    else:
        return "Not a greeting"

def predict(message, bot, assistant):
    '''
    Purpose: to generate a response for chatbot interface
    @params message: user input/prompt (str)
    @params bot: the model that the users selects as the bot assistant in the chatbot UI (str)
    @params assistant: the selected assistant (aka. model) from model_select dropdown (str)
    returns: the model generated answer for a given question/message
    '''
    # assistant to model mapping
    model_mapping = {
        "Flo":"flan-t5"
        ,"Biscuit":"falcon-7b"
        ,"Lois":"llama-7b"
        ,"Bubba":"gpt3.5"
    }

    try:
        model_select = model_mapping[assistant]
    except:
        model_select = "no model"

    if model_select == "no model":
        return "Select an assistant to start asking questions."

    # check if message/query is a greeting
    greet_msg = greet(message, assistant)
    if greet_msg != "Not a greeting":
        return greet_msg
    else:
        # send message to the right model
        if model_select == "flan-t5":
            ans = f"{assistant}: {flanT5_predict(flan_t5_pipe, message)}"
            # ans = f"{assistant}: some response here"

        elif model_select == "falcon-7b":
            ans = f"{assistant}: {falcon_predict(falcon_peft_model, falcon_peft_tokenizer, message)}"
            # ans = f"{assistant}: some response here"

        elif model_select == "llama-7b":
            ans = f"{assistant}: {llama2_predict(llama_pipe, message)}"
            # ans = f"{assistant}: some response here"

        elif model_select == "gpt3.5":
            ans = f"{assistant}: {gpt_predict(message)}"
            # ans = f"{assistant}: some response here"

        else:
            ans = "Select an assistant to start asking questions."
            
        # insert question-answer generated data to mongodb
        data = {'datetime': datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
                ,'question':message
                ,'response':ans
            }
        
        db = 'GRADIO_APP'
        coll = 'QA_GENERATED'
        resp = insert_data(data, db, coll)

        return ans


# set up the chatbot application interface
with gr.Blocks() as get_started_block:
    gr.HTML(
        """
        <h2>Project Inspiration</h2>
        Increasingly more people have become dog parents since the 2019 pandemic. But vet bills are expensive, so chatbots to answer dog-related questions were created with this project. 
        With LLMs such as ChatGPT on the rise, this project aims to compare and evaluate whether open-source LLMs can perform as well as private LLMs for specific domains.
        So this project chose dog wellness as the topic to evaluate as such.
        <h2>What to do</h2>
        <ol>
          <li>Go to the "Start Asking" tab.</li>
          <li>Choose a dog assistant; each assistant corresponds to a LLM.</li>
          <li>Start asking questions.</li>
        </ol>
        <h2>Help us improve by</h2>
        --> Hitting the "thumbs up" button if you like a response. If not, hit the "thumbs down".<br>
        (All responses are recorded and reflected in the <i>Assistant Ratings</i> tab.)
        <h3>Tip: You can toggle between different dog assistants mid-conversation to compare assistant responses.</h3>
        """
    )

with gr.Blocks() as chat_block:
    assistant = gr.Dropdown(["Biscuit", "Bubba", "Flo", "Lois"]  # assistant names for each LLM
                            ,label = "Select a dog assistant"
                            ,max_choices = 1
                            ,multiselect = False)

    chatbot_component = gr.Chatbot(render=False
                                   ,likeable=True
                                   ,height=500)

    chatbot = gr.ChatInterface(predict
                               ,additional_inputs = [assistant]
                               ,chatbot = chatbot_component)

    chatbot_component.like(print_like_dislike, None, None)


with gr.Blocks() as rating_block:
    with gr.Row(equal_height=False):
        gr.HTML(
            """
            <center>
            <h1>Assistant Ratings</h1>
            See how others rated each assistant.<br>
            This page shows the following info for each assistant: 
            total # of questions answered, total # of answers generated that were liked/disliked, and the % of total answers liked.<br><br>
            <b>Click on the "Refresh ratings" button at the bottom to refresh rating results.</b>
            <h2>The assistant with the most liked responses is..</h2>
            </center>
            """
        )
    
    with gr.Row(equal_height=False):
        gradio_label = gr.Label(value=create_labels()
                                ,show_label=False
                               )

    refresh_button = gr.Button("Refresh ratings", rating_block)
    refresh_button.click(create_labels, [], gradio_label)

    
with gr.Blocks(theme='JohnSmith9982/small_and_pretty') as demo:
    gr.HTML(
        """
        <table border=0>
          <tr>
            <td><img src="https://i.pinimg.com/originals/c7/89/60/c78960079a1973c82df58f16c76308d8.gif" width="100", height="50"></td>
            <td><h1>Project Marley: a Dog Care AI</h1>Save time and money by getting quick answers on any dog health and wellness questions.</td>
          </tr>
        </table>
        """
    )
    
    gr.TabbedInterface([get_started_block, chat_block, rating_block], ["How to Get Started", "Start Asking", "Assistant Ratings"])

demo.queue(default_concurrency_limit=5, max_size=40)
demo.launch()