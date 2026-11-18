# RAG: Vector Databases with ChromaDB Certification


## What this project demonstrates

I’m Ashwik, an aspiring AI/ML architect. Through these hands-on labs, I learned how text becomes searchable vector embeddings, how distance metrics affect retrieval, and how structured metadata can make semantic search more targeted. I connected the mathematics of similarity search with practical ChromaDB examples using grocery descriptions and employee profiles.

For recruiters and hiring teams, this repository demonstrates my practical skills in:

- **Python and ChromaDB:** creating collections, adding documents with IDs and metadata, and querying embedded text.
- **SentenceTransformers and semantic search:** generating document and query embeddings, ranking matches, and interpreting similarity scores.
- **Vector mathematics:** implementing Euclidean distance, dot products, normalization, and cosine similarity, then checking manual calculations against NumPy, SciPy, and PyTorch.
- **Metadata filtering:** using department, experience, and location filters, including `$and`, `$gte`, and `$in`, alongside semantic queries.
- **RAG retrieval fundamentals:** understanding how embedding generation, vector storage, and retrieval support a larger RAG system.
- **Technical communication and experimentation:** documenting runnable examples and using a Jupyter lab to explore how changes to documents, queries, and result counts affect retrieval.

These learning exercises demonstrate foundational proficiency in vector search and give me a practical base for developing more advanced AI applications.

## Repository overview

Python examples that use Chroma DB and SentenceTransformers for semantic search over grocery descriptions and employee profiles. Learn how to store documents, generate embeddings, retrieve similar results, and combine similarity search with metadata filters.

This repository demonstrates vector storage and retrieval, a building block of retrieval-augmented generation (RAG). It focuses on retrieval and does not include an LLM answer-generation step.

## Files

- [Similarity Search by Hand.ipynb](Similarity%20Search%20by%20Hand.ipynb): beginner-friendly, hands-on learning lab for embeddings, Euclidean distance, dot products, cosine similarity, and ranked query search.
- [`ChromaDB_query.py`](ChromaDB_query.py): grocery collection creation, sample data, and similarity search.
- [`similarity_employeedata.py`](similarity_employeedata.py): employee profile similarity search, metadata filtering, and combined queries.
- [`requirements-notebook.txt`](requirements-notebook.txt): shared dependencies plus NumPy, SciPy, and JupyterLab.
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

Install the shared dependencies:

```bash
python -m pip install -r requirements.txt
```

The requirements specify `chromadb==1.0.12`, `sentence-transformers==4.1.0`, and an unpinned `torch` dependency, with the PyTorch CPU wheel index configured as an extra package source. Allow internet access on the first run so the embedding model can download.

Run either example independently:

```bash
# Grocery similarity search
python ChromaDB_query.py

# Employee similarity search and metadata filtering
python similarity_employeedata.py
```

Both examples create an in-memory Chroma client, use the `all-MiniLM-L6-v2` embedding model, and configure cosine distance. Their collections are recreated on each run and are not saved to disk.

## Grocery similarity search

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

## Employee similarity search and metadata filtering

[`similarity_employeedata.py`](similarity_employeedata.py) creates `Employee_collection` with 15 sample employee profiles across Engineering, Marketing, and HR.

Each profile becomes a document describing the employee's role, years of experience, department, skills, location, and employment type. Each document has a unique ID (`employee_1` through `employee_15`) and structured metadata for the employee's name, department, role, experience, location, and employment type.

The `perform_advaced_search(collection, all_items)` function runs these examples:

| Search | Method | Criteria |
| --- | --- | --- |
| Python developers | `collection.query()` | Top 3 matches for `Python developer with web development experience` |
| Leadership and management | `collection.query()` | Top 3 matches for `team leader manager with experience` |
| Engineering employees | `collection.get()` | Department equals `Engineering` |
| Experienced employees | `collection.get()` | Experience is at least 10 years (`$gte: 10`) |
| Employees in California cities | `collection.get()` | Location is `San Francisco` or `Los Angeles` (`$in`) |
| Senior Python developers in selected cities | `collection.query()` | Up to 5 matches, restricted to at least 8 years of experience and locations in `San Francisco`, `New York`, or `Seattle` |

Metadata retrieval with `collection.get()` selects records by their stored fields. Similarity queries with `collection.query()` rank documents by embedding distance. Combining both lets you constrain the candidate profiles before ranking them semantically.

For example, the combined search uses `$and`, `$gte`, and `$in`:

```python
results = collection.query(
    query_texts=["senior Python developer full-stack"],
    n_results=5,
    where={
        "$and": [
            {"experience": {"$gte": 8}},
            {"location": {"$in": ["San Francisco", "New York", "Seattle"]}},
        ]
    },
)
```

The script prints matching employee details and distances for similarity queries, and employee details for metadata lookups. Lower cosine distances indicate closer matches.

## Customize the examples

### Grocery queries

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


### Employee queries

Edit the `employees` list in `main()` to change the sample profiles. Modify the query text and `where` filters in `perform_advaced_search()` to explore other roles, experience thresholds, departments, and locations. Both examples use the same `requirements.txt`; no additional dependencies are needed for the employee example.


## Similarity search by hand notebook

The [Similarity Search by Hand](Similarity%20Search%20by%20Hand.ipynb) notebook is part of a learning exercise and hands-on lab for beginners. It introduces foundational concepts such as embeddings, vector distances, normalization, and similarity search through code you can run, inspect, and modify one cell at a time. The completed exercises help connect the underlying mathematics to practical Python implementations.

Use the lab to experiment: change the sample documents or query text, predict which results should be closest, and compare your expectations with the calculated scores. It explores vector search without a vector database and uses `paraphrase-MiniLM-L6-v2` to embed five documents, including Ashwik's software debugging example.

```bash
python -m pip install -r requirements-notebook.txt
jupyter lab "Similarity Search by Hand.ipynb"
```

Select your virtual environment's Python kernel and use **Restart Kernel and Run All Cells**. The first run requires internet access to download the model.

The exercises implement an efficient symmetric Euclidean distance matrix, verify unit-length normalization, and rank documents by cosine similarity to a query. Numerical assertions compare manual calculations with SciPy, NumPy, and PyTorch results.

To customize the lab, edit `documents` and rerun the notebook, or change `query_text` and `top_k` in Exercise 3. Matrix sizes adapt to the document count. The original Skills Network / IBM author credits and copyright notice are retained in the notebook.

## Tags

`python` · `chromadb` · `vector-database` · `semantic-search` · `similarity-search` · `text-embeddings` · `sentence-transformers` · `rag` · `retrieval-augmented-generation` · `cosine-similarity` · `euclidean-distance` · `metadata-filtering` · `numpy` · `scipy` · `pytorch` · `jupyter-notebook` · `machine-learning` · `hands-on-learning`
