# RAG: Vector Databases with ChromaDB Certification

Python examples that use Chroma DB and SentenceTransformers for semantic search over grocery descriptions and employee profiles. Learn how to store documents, generate embeddings, retrieve similar results, and combine similarity search with metadata filters.

This repository demonstrates vector storage and retrieval, a building block of retrieval-augmented generation (RAG). It focuses on retrieval and does not include an LLM answer-generation step.

## Files

- [`ChromaDB_query.py`](ChromaDB_query.py): grocery collection creation, sample data, and similarity search.
- [`similarity_employeedata.py`](similarity_employeedata.py): employee profile similarity search, metadata filtering, and combined queries.
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
