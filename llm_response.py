import logging

from huggingface_hub import hf_hub_download
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.llms import LlamaCpp
from langchain.vectorstores import Milvus

from constants import (EMBEDDING_MODEL_NAME, MODEL_ID, MODEL_BASENAME)

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)s - %(message)s",
                    level=logging.INFO)


def load_cpu_model(model_id, model_basename):
    model_path = hf_hub_download(repo_id=model_id, filename=model_basename)

    # Callbacks support token-wise streaming
    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
    # Verbose is required to pass to the callback manager

    model = LlamaCpp(model_path=model_path, n_ctx=2048, max_tokens=2048, temperature=0, repeat_penalty=1.15,
                     callback_manager=callback_manager, verbose=True)

    return model


# Load the embedding model
embeddings = HuggingFaceInstructEmbeddings(model_name=EMBEDDING_MODEL_NAME, model_kwargs={"device": 'cuda'})
logging.info('embedding model loaded.')

# Load text generator model
llm = load_cpu_model(MODEL_ID, MODEL_BASENAME)
logging.info('text generator model loaded.')


def get_llm_generation_langchain(question):
    vector_store = Milvus(embedding_function=embeddings,
                          connection_args={"host": "localhost", "port": "19530"},
                          collection_name='PAN',
                          index_params={"metric_type": "IP", "params": {"nprobe": 10}})

    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=vector_store.as_retriever(),
                                     return_source_documents=True)

    # Get the answer from the chain
    res = qa(question)
    answer, docs = res["result"], res["source_documents"]

    return answer
