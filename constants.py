import os

ROOT_DIRECTORY = os.path.dirname(os.path.realpath(__file__))

# Define the folder for storing database
DB_BASE_DIRECTORY = 'milvus-data'
DB_COLLECTION_NAME = 'PAN'

# Define Data directory
DATA_SOURCE_DIRECTORY = os.path.join(ROOT_DIRECTORY, 'data')

# Default Instructor Model
EMBEDDING_MODEL_NAME = "hkunlp/instructor-large"

# LLM model - for cpu only
MODEL_ID = "TheBloke/wizard-vicuna-13B-GGML"
MODEL_BASENAME = "wizard-vicuna-13B.ggmlv3.q4_0.bin"

