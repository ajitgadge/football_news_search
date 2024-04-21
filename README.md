# football_news_search

==============================================================================

I have downloaded the data set about football news from https://www.kaggle.com/. You can find the data in the .csv file here allfootball_new.csv. 
This is text-based content. 
I like to search for football news based on Retrieval-Augmented Generation (RAG) to provide human-like text. I am using PostgreSQL pg_vector functionality to store the data in text and then embed it into PostgreSQL in the vector data type using a Python-based program. Once data is stored, tokenised and embedded, I have created a Python-based programme that queries pg_vector's Euclidean distance and L2 distance similarity search functionality of pg_vector. Here are the basic steps.

**Step 1: **
Ingest the .csv file batch-wise and, simultaneously, tokenise the same and then embed it into vector form into the PostgreSQL database using vector as data type.
I have used the Python-based sentence transform model Hugging face model  'all-MiniLM-L6-v2'. It maps sentences and paragraphs to 384 dimensions.

 # Initialize the model
model = SentenceTransformer('all-MiniLM-L6-v2').

Using this model, I have created a vector as a return to ingest into the PostgreSQL table.

def preprocess_and_vectorize(documents):
    # Preprocess (e.g., tokenisation, cleaning) and vectorise documents
    vectors = model.encode(documents, show_progress_bar=True)
    return vectors

You can find the complete Python programme here: file_ingest_football_news.py.

I have ingested this paragraph in batches so that performance is better. 

**Step 2:  **
Once data is ingested into the table and vectorised using the model , I need to search the relevant data with context. 
I am taking input from my command prompt as questions and then trying to search the relevant content and text from the tables that I have ingested. But I don’t want to search in text; instead, I want to search with vector data to get a similarity search. 

To do so, I take input from the command prompt as text and then convert that text input into vector form to do a similarity search against my stored vector data. I am using the Hugging face model  'all-MiniLM-L6-v2' for this.

I have to search the vector data using Euclidean distance <->. I can take the results for the top 1, 2, or so (top_k) similarity.

SELECT content FROM documents ORDER BY vector <-> %s::vector LIMIT %s", (vector, top_k).

Once I get these results, I would like to combine these similar search results into human-like text to get a human-like answer.

I am using Generator OpenAI GPT-2 large language model to combine and generate my answers based on my similarity search data.

def generate_answer(documents):
    # Combine the documents into a single text context
    context = " ".join(documents)
    # Generate an answer based on the combined context
    answers = generator(context, max_length=50000, num_return_sequences=1, truncation=True)
    return answers[0]['generated_text']

I am restricting my answer for max_lenght so it will not go out of token length. You can find the complete programme here: rag_search_football_news.py.

Here is how this works in Semantic search with context searching.

 <img width="452" alt="image" src="https://github.com/ajitgadge/football_news_search/assets/35986148/962a54eb-b76e-4348-ac01-c2297447f407">
 

Some of the examples that I ran are below.

Example 1: 

 <img width="452" alt="image" src="https://github.com/ajitgadge/football_news_search/assets/35986148/322fdbb8-80a8-4cf1-a1dc-543f51f19c76">



Example 2:

<img width="452" alt="image" src="https://github.com/ajitgadge/football_news_search/assets/35986148/d21b2017-dc9e-4d9e-a1bc-3666a3cf8ac8">


 





![image](https://github.com/ajitgadge/football_news_search/assets/35986148/af685d2a-ad2b-45b6-92d5-a6a283bbacdb)
