import logging
import os
import subprocess

from langchain.document_loaders import TextLoader
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Milvus
from milvus import default_server

from constants import (DB_BASE_DIRECTORY,
                       DB_COLLECTION_NAME,
                       DATA_SOURCE_DIRECTORY,
                       EMBEDDING_MODEL_NAME)

# Load the embedding model
embeddings = HuggingFaceInstructEmbeddings(model_name=EMBEDDING_MODEL_NAME, model_kwargs={"device": 'cuda'})


# Create an embedding for given text/doc and insert it into Milvus Vector DB
def get_langchain_format_docs(data_path):
    loader = TextLoader(os.path.join(data_path, 'KnowledgeDocument.txt'), encoding='utf8')
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=400,
                                                   chunk_overlap=20,
                                                   separators=["\n\n", "\n", " ", ""])
    docs = text_splitter.split_documents(docs)

    return docs


def main():
    # Reset the vector database files
    logging.info("Reset the vector database files")
    print(subprocess.run(["rm -rf milvus-data"], shell=True))

    default_server.set_base_dir(DB_BASE_DIRECTORY)
    default_server.start()
    try:
        docs = get_langchain_format_docs(f'{DATA_SOURCE_DIRECTORY}')
        vector_store = Milvus.from_documents(docs,
                                             embeddings,
                                             collection_name=f"{DB_COLLECTION_NAME}")

        logging.info("Milvus database is up and collection is created")
        logging.info("Finished loading Knowledge Base embeddings into Milvus")

    except Exception as e:
        logging.info(e)
        default_server.stop()
        raise e

    default_server.stop()


if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)s - %(message)s",
                        level=logging.INFO)
    main()
