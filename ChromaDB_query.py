# Grocery similarity search with Chroma DB and SentenceTransformers.
# Stores 14 sample grocery descriptions with IDs and metadata in an in-memory
# collection, then searches for "apple" and "meat" and displays three matches each.
# Scores are cosine distances: lower values indicate closer matches.
#
# Run: python ChromaDB_query.py (after installing requirements.txt).
# Customize:
# - In perform_similarity_search(), edit query_term, keeping it a list of strings:
#   query_term = ["coffee", "fresh fruit"]
# - Add n_results=3 to collection.query() to request three matches per query.
#   The current query uses Chroma's default count; the display loop shows three.
#   To display a different count, also update min(3, ...) and the "Top 3" heading.
# - In main(), edit texts to change the dataset. IDs and metadata are generated
#   automatically from that list; customize metadatas to add your own labels.
# - Change the model name in ef to try another SentenceTransformer model.
# Data is recreated each run; the first run may download the embedding model.

import chromadb
from chromadb.utils import embedding_functions

ef = embedding_functions.SentenceTransformerEmbeddingFunction("all-MiniLM-L6-v2")

client = chromadb.Client()

collection_name = "my_grocery_collection"

def main():
    try:
        collection = client.create_collection(
            name = collection_name,
            metadata = {"description":"A collection for storing my grocery data."},
            configuration={
                "hnsw":{"space":"cosine"},
                "embedding_function" : ef
            }
        )
        print(f"Collection created: {collection_name}")
        texts=[
            'fresh red apples',
            'organic bananas',
            'ripe mangoes',
            'whole wheat bread',
            'farm-fresh eggs',
            'natural yogurt',
            'frozen vegetables',
            'grass-fed beef',
            'free-range chicken',
            'fresh salmon fillet',
            'aromatic coffee beans',
            'pure honey',
            'golden apple',
            'red fruit'
        ]
        ids = [f"food_{index+1}" for index, _ in enumerate(texts)]

        collection.add(
            documents=texts,
            metadatas=[{"source":"grocery_store","category":"food"} for _ in texts],
            ids = ids
        )

        all_items = collection.get()
        # Log the retrieved items to the console for inspection
        # This will print out all the documents, IDs, and metadata stored in the collection
        print("Collection contents:")
        print(f"Number of documents: {len(all_items['documents'])}")
        perform_similarity_search(collection, all_items)

        pass
    except Exception as error:
        print(f"Error:{error}")

def perform_similarity_search(collection, all_items):
    try:
        query_term = ["apple","meat"]
        results = collection.query(
            query_texts=query_term
        )
        print(f"Query results for '{query_term}':")
        print(results)

        # Check if no results are returned or if the results array is empty
        if not results or not results['ids'] or len(results['ids'][0]) == 0:
            # Log a message indicating that no similar documents were found for the query term
            print(f'No documents found similar to "{query_term}"')
            return
        
        # Access the nested arrays in 'results["ids"]' and 'results["distances"]'
        for q in range(len(results['ids'])):
            print("\n","+" * 50)
            print("+" * 50,"\n")
            print(f'Top 3 similar documents to "{query_term[q]}":')
            for i in range(min(3, len(results['ids'][0]))):
                doc_id = results['ids'][q][i]  # Get ID from 'ids' array
                score = results['distances'][q][i]  # Get score from 'distances' array
                # Retrieve text data from the results
                text = results['documents'][q][i]
                if not text:
                    print(f' - ID: {doc_id}, Text: "Text not available", Score: {score:.4f}')
                else:
                    print(f' - ID: {doc_id}, Text: "{text}", Score: {score:.4f}')


        pass
    except Exception as error:
        print(f"Error in similarity search: {error}")

if __name__ == "__main__":
    main()
