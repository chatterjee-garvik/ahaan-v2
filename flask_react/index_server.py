import os
from dotenv import load_dotenv

load_dotenv()

os.environ['OPENAI_API_KEY'] = "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"
print(os.environ['OPENAI_API_KEY'])
print(os.environ['PINECONE_API_KEY'])
print(os.environ['PINECONE_ENVIRONMENT'])
print(os.environ['PINECONE_INDEX'])


import pickle

# NOTE: for local testing only, do NOT deploy with your key hardcoded
# os.environ['OPENAI_API_KEY'] = "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"


from multiprocessing import Lock
from multiprocessing.managers import BaseManager
from llama_index import SimpleDirectoryReader, GPTVectorStoreIndex, Document, ServiceContext, StorageContext, load_index_from_storage
from llama_index.vector_stores import PineconeVectorStore
import pinecone

api_key = os.environ['PINECONE_API_KEY']
pinecone.init(api_key=api_key, environment=os.environ['PINECONE_ENVIRONMENT'])
pinecone_index = pinecone.Index(os.environ['PINECONE_INDEX'])

index = None
stored_docs = {}
lock = Lock()

# index_name = "./saved_index"
pkl_name = "stored_documents.pkl"
# api_key = "abebac91-f48a-4191-9308-61c28d58f223"

def initialize_index():
    """Create a new global index, or load one from the pre-set path."""
    global index, stored_docs
    vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    with lock:
        index = GPTVectorStoreIndex.from_documents([], storage_context=storage_context)
        if os.path.exists(pkl_name):
            with open(pkl_name, "rb") as f:
                stored_docs = pickle.load(f)


def query_index(query_text):
    """Query the global index."""
    global index
    response = index.as_query_engine().query(query_text)
    return response


def insert_into_index(doc_file_path, doc_id=None):
    """Insert new document into global index."""
    global index, stored_docs
    document = SimpleDirectoryReader(input_files=[doc_file_path]).load_data()[0]
    if doc_id is not None:
        document.doc_id = doc_id

    with lock:
        # Keep track of stored docs -- llama_index doesn't make this easy
        stored_docs[document.doc_id] = document.text[0:200]  # only take the first 200 chars keep

        index.insert(document) # keep
        # index.storage_context.persist(persist_dir=index_name)
        
        with open(pkl_name, "wb") as f:
            pickle.dump(stored_docs, f)

    return

def get_documents_list():
    """Get the list of currently stored documents."""
    global stored_doc
    documents_list = []
    for doc_id, doc_text in stored_docs.items():
        documents_list.append({"id": doc_id, "text": doc_text})

    return documents_list


if __name__ == "__main__":
    # init the global index
    print("initializing index...")
    initialize_index()

    # setup server
    # NOTE: you might want to handle the password in a less hardcoded way
    manager = BaseManager(('', 5602), b'password')
    manager.register('query_index', query_index)
    manager.register('insert_into_index', insert_into_index)
    manager.register('get_documents_list', get_documents_list)
    server = manager.get_server()

    print("server started...")
    server.serve_forever()
