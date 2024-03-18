import os
from langchain.llms.bedrock import Bedrock
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain

def demo_chatbot():
    demo_llm = Bedrock(
        model_id='meta.llama2-70b-chat-v1',
        model_kwargs={
            "prompt":"string",
            "temperature":0.9,
            "top_p":0.5,
            "max_gen_len":512
        }
    )
    return demo_llm

# create a function for Conversation Chain - Input text + Memory
def demo_memory():
    llm_data=demo_chatbot()
    memory = ConversationBufferMemory(llm=llm_data,max_token_limit=512)
    return memory

# Create a Function for Conversation Chain - Input Text + Memory
def demo_conversation(input_text,memory):
    llm_chain_data = demo_chatbot()
    llm_conversation= ConversationChain(llm=llm_chain_data,memory=memory,verbose=True)

# Chat response using Predict (Prompt template)
    chat_reply = llm_conversation.predict(input=input_text)
    return chat_reply





# temperature decide the randomness of the response that you're going to get from the foundation model
# top_p defining the randomness and diversity of the response 
# every 100 token is equal to 75 words approx.