# Program 1: Ingesting Documents into PostgreSQL

import pandas as pd
from sentence_transformers import SentenceTransformer
import psycopg2
import psycopg2.extras

# Configuration
DATABASE_URI = "postgresql://postgres:xxxxx@xxxxx/blog_data"
BATCH_SIZE = 1000
VECTOR_DIMENSION = 384
CSV_FILE_PATH = "/Users/ajitxxx/Documents/xxxxx/Kaggel_DataSet/allfootball_new.csv"  # or "path_to_your_file.txt"

# Initialize the model
model = SentenceTransformer('all-MiniLM-L6-v2')

def preprocess_and_vectorize(documents):
    # Preprocess (e.g., tokenization, cleaning) and vectorize documents
    vectors = model.encode(documents, show_progress_bar=True)
    return vectors

def batch_insert(documents, vectors):
    conn = psycopg2.connect(DATABASE_URI)
    cursor = conn.cursor()
    
    for doc, vec in zip(documents, vectors):
        # Ensure vec is a NumPy array, then convert to a list of Python floats
        vec_list = [float(item) for item in vec]  # Explicit conversion
        
        cursor.execute("INSERT INTO documents (content,vector) VALUES (%s, %s)", (doc, vec_list))
    
    conn.commit()
    cursor.close()
    conn.close()

def ingest_documents(file_path):
    # Load documents from a CSV or text file
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
        documents = df.iloc[:, 0].astype(str).tolist()  # Assuming text is in the first column
    else:
        with open(file_path, 'r') as file:
            documents = file.readlines()

    # Process documents in batches
    for i in range(0, len(documents), BATCH_SIZE):
        batch_docs = documents[i:i+BATCH_SIZE]
        vectors = preprocess_and_vectorize(batch_docs)
        batch_insert(batch_docs, vectors)
        print(f"Batch {i//BATCH_SIZE + 1} inserted.")

if __name__ == "__main__":
    ingest_documents(CSV_FILE_PATH)

