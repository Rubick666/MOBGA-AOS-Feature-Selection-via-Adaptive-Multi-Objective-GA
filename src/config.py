import os
from dotenv import load_dotenv

load_dotenv()

# -----------------------------
# Experiment Configuration
# -----------------------------
DATASETS = ["DS02", "DS04", "DS05", "DS07", "DS08", "DS10"]
DATASET_DIR = os.getenv("DATASET_DIR", "Datasets")

NUM_RUNS = int(os.getenv("NUM_RUNS", 3))
GENERATIONS = int(os.getenv("GENERATIONS", 50))
POP_SIZE = int(os.getenv("POP_SIZE", 50))
MUTATION_RATE = float(os.getenv("MUTATION_RATE", 0.01))