import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')

# TwiBot-22 specific file paths (for full dataset)
TWIBOT_GLOBAL_DATA_PATHS = {
    'user_json': os.path.join(RAW_DATA_DIR, 'user.json'),
    'label_csv': os.path.join(RAW_DATA_DIR, 'label.csv'),
    'edge_csv': os.path.join(RAW_DATA_DIR, 'edge.csv'),
}

