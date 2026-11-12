import os
import numpy as np
import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer

def main():
    print("1. Initializing ChromaDB and Embedding Model...")
    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_or_create_collection(name="document_collection")
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # 2. Reading text documents from Dataset/documents folder
    dataset_path = "Dataset"
    target_dir = dataset_path
    
    if os.path.exists(os.path.join(dataset_path, "documents")):
        target_dir = os.path.join(dataset_path, "documents")

    documents = []
    metadatas = []
    ids = []

    if os.path.exists(target_dir):
        for i, filename in enumerate(os.listdir(target_dir)):
            if filename.endswith(".txt"):
                file_path = os.path.join(target_dir, filename)
                with open(file_path, "r", encoding="utf-8") as f:
                    text = f.read()
                    documents.append(text)
                    metadatas.append({
                        "title": filename,
                        "category": "AI / Technology",
                        "source": file_path
                    })
                    ids.append(f"doc_{i+1}")

    if len(documents) > 0:
        print(f"Found {len(documents)} documents. Storing into ChromaDB...")
        collection.upsert(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        
        # 3. Automatically generate and save embeddings.npy inside Dataset folder
        embeddings = model.encode(documents)
        embeddings_path = os.path.join(dataset_path, "embeddings.npy")
        np.save(embeddings_path, embeddings)
        print(f"[SUCCESS] embeddings.npy created at: {embeddings_path}")
        
        print("Successfully stored/updated all document embeddings!")
    else:
        print(f"Warning: No .txt files found inside '{target_dir}'.")
        return

    # 4. Define sample queries and automatically create queries.csv inside Dataset folder
    queries = [
        "What is artificial intelligence?",
        "How do vector embeddings work in databases?",
        "Explain semantic search concepts.",
        "What is the role of ChromaDB in LLMs?",
        "How to generate sentence embeddings?"
    ]

    queries_data = [{"query_id": f"Q{i+1}", "query_text": q} for i, q in enumerate(queries)]
    df_queries = pd.DataFrame(queries_data)
    queries_csv_path = os.path.join(dataset_path, "queries.csv")
    df_queries.to_csv(queries_csv_path, index=False)
    print(f"[SUCCESS] queries.csv automatically created at: {queries_csv_path}")

    # 5. Perform Similarity Search
    print("\n--- Running Similarity Search for Queries ---")
    results = collection.query(
        query_texts=queries,
        n_results=min(5, len(documents))
    )

    # 6. Automatically create retrieved_results.csv inside Dataset folder
    output_data = []
    for q_idx, query in enumerate(queries):
        matched_docs = results['documents'][q_idx]
        matched_metas = results['metadatas'][q_idx]
        for m_idx, (doc, meta) in enumerate(zip(matched_docs, matched_metas)):
            output_data.append({
                "Query": query,
                "Rank": m_idx + 1,
                "Document_Title": meta.get("title"),
                "Category": meta.get("category"),
                "Content_Snippet": doc[:200]
            })

    df_results = pd.DataFrame(output_data)
    csv_output_path = os.path.join(dataset_path, "retrieved_results.csv")
    df_results.to_csv(csv_output_path, index=False)
    print(f"[SUCCESS] retrieved_results.csv automatically created at: {csv_output_path}")

if __name__ == "__main__":
    main()