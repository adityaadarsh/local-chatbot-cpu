import os

import gradio
from langchain.callbacks import wandb_tracing_enabled
from milvus import default_server

from llm_response import get_llm_generation_langchain

os.environ["LANGCHAIN_WANDB_TRACING"] = "true"
os.environ["WANDB_PROJECT"] = "langchain-testing"


def main():
    # Configure gradio QA app
    print("Configuring gradio app")
    demo = gradio.Interface(fn=get_responses,
                            inputs=gradio.Textbox(label="Question", placeholder=""),
                            outputs=[gradio.Textbox(label="Asking LLM with Context (RAG)")],
                            examples=["What is the cost/fees of a PAN card?",
                                      "How to apply for PAN card"],
                            allow_flagging="never")

    # Launch gradio app
    print("Launching gradio app")
    demo.launch(share=True,
                enable_queue=True,
                show_error=True,
                server_name='127.0.0.1',
                server_port=80)
    print("Gradio app ready")


# Helper function for generating responses for the QA app
def get_responses(question):
    # Load Milvus Vector DB collection
    # enable tracing using a context manager
    with wandb_tracing_enabled():
        return get_llm_generation_langchain(question)


if __name__ == "__main__":
    # Start Milvus Vector DB
    default_server.stop()
    default_server.set_base_dir('milvus-data')
    default_server.start()

    main()
