import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


# Configuration
DATA_PATH = os.getenv("DATA_PATH")
MODEL_OUTPUT_PATH = os.getenv("MODEL_OUTPUT_PATH")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")


class Config:
    SEED = 42
    EPOCHS = 100
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    MODEL_TYPE = "RandomForest"
