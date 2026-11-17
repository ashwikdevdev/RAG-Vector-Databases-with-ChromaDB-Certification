# RAG: Vector Databases with ChromaDB Certification

A Python example that uses Chroma DB and SentenceTransformers to search grocery descriptions by meaning. The script stores sample documents with metadata, generates embeddings, and displays the top three matches for each query.

This repository demonstrates vector storage and retrieval, a building block of retrieval-augmented generation (RAG). It focuses on retrieval and does not include an LLM answer-generation step.

## Files

- [`ChromaDB_query.py`](ChromaDB_query.py): collection creation, sample data, and similarity search.
- [`requirements.txt`](requirements.txt): Python dependencies and the PyTorch CPU package index.

## Setup and usage

Open a terminal in the folder containing the script and `requirements.txt`. Create and activate a virtual environment:

```bash
python -m venv .venv
```

On macOS or Linux:

```bash
source .venv/bin/activate
```

On Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Install the dependencies and run the example:

```bash
python -m pip install -r requirements.txt
python ChromaDB_query.py
```

The requirements specify `chromadb==1.0.12`, `sentence-transformers==4.1.0`, and an unpinned `torch` dependency, with the PyTorch CPU wheel index configured as an extra package source. Allow internet access on the first run so the embedding model can download.

## How it works

### Chroma DB integration

The code imports `chromadb` and its embedding utilities, creates a client with `chromadb.Client()`, and configures a SentenceTransformer embedding function using `all-MiniLM-L6-v2`. The collection configuration specifies cosine distance for comparing embeddings.

### Data handling

The script creates a collection named `my_grocery_collection` with a description of the grocery dataset. It adds 14 sample text documents, including `fresh red apples`, `organic bananas`, `grass-fed beef`, and `golden apple`.

Each document receives:

- A unique ID, from `food_1` through `food_14`.
- Metadata containing `source: grocery_store` and `category: food`.
- An embedding generated automatically by the configured SentenceTransformer embedding function when the document is added.

After insertion, the script retrieves the collection contents and prints the number of documents. The client uses an in-memory collection; this example does not save the data between runs.

### Similarity search

The `perform_similarity_search(collection, all_items)` function queries the collection using text input. The current example searches for both `apple` and `meat`:

```python
query_term = ["apple", "meat"]
results = collection.query(query_texts=query_term)
```

For each query, the function iterates through the results and displays the top three matching documents with their IDs, text, and scores. For `apple`, this lets you compare apple-related descriptions with other grocery items based on their embeddings.

The printed `Score` is a **cosine distance**, not a similarity percentage. **Lower scores indicate closer matches.** The query uses Chroma's default result count, and the display loop limits the output to three matches per query. The script also prints the raw query results before the formatted matches.

## Customize the example

To search only for apples, change the query list inside `perform_similarity_search`:

```python
query_term = ["apple"]
```

To request exactly three results per query from Chroma, add `n_results=3`:

```python
results = collection.query(
    query_texts=query_term,
    n_results=3,
)
```

Edit the `texts` list in `main()` to try different grocery descriptions. The script generates the document IDs and metadata entries from that list automatically.
