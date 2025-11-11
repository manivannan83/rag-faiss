import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent / "src"))
from src.data_loader import load_all_documents
from src.vectorstore import FaissVectorStore
from src.search import RAGSearch


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Expected 2 arguments which is user prompt and llm model.")
        sys.exit(1)
    
    user_prompt = sys.argv[1]
    llm_model = sys.argv[2]

    # project_root
    project_root = Path(__file__).resolve().parent
    # db_dir
    db_dir = project_root / "db/faiss_store"
    
    rag_search = RAGSearch(persist_dir=db_dir, llm_model=llm_model)
    summary = rag_search.search_and_summarize(query=user_prompt, top_k=3)
    print("Summary:", summary)
