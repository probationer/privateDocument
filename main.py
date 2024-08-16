import os
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(), override=True)

def load_pdf(file):
    from langchain.document_loaders import PyPDFLoader
    print(f' Loading {file} ...')
    loader = PyPDFLoader(file)
    data = loader.load()
    return data

# data = load_pdf('1212.5701.pdf')
# print(data)
# print(data[1].page_content)
# print(len(data))

def load_document(file):
    import os 
    name, extension = os.path.splitext(file)
    if extension == '.pdf': 
        from langchain.document_loaders import PyPDFLoader
        print(f' Loading {file} ...')
        loader = PyPDFLoader(file)
    elif extension == '.docx':
        from langchain.document_loaders import Docx2txtLoader
        print(f' Loading {file} ...')
        loader = Docx2txtLoader(file)
    else:
        print('Document format is not supported! ')

    data = loader.load()
    return data

def load_from_wikipedia(query, lang='en', load_max_docs=2):
    from langchain.document_loaders import WikipediaLoader
    loader = WikipediaLoader(query=query, lang=lang, load_max_docs=load_max_docs)
    data = loader.load()
    return data

# data = load_from_wikipedia('GPT-4')
# print(data[0].page_content)

# 2. Chunking 
# It is the process of breaking down large pieces of text into smaller segments.
# it's an enssential technique that helps optimize the relevance of the content we get back from a vector database
# As a rule of thumb, if a chunk of text makes sense without the surrounding context to a human, it will make a sense to the language model as well.
# Finding the optimal chunk size for the documents in the corpus is crucial to ensure that the search result are accurate and relevant.

def chunk_data(data, chunk_size=256):
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=0)
    chunks = text_splitter.split_documents(data)
    return chunks

# data = load_pdf('1212.5701.pdf')
# chunks = chunk_data(data)
# print(len(chunks))
# print(chunks[1].page_content)

def print_embedding_cost(texts):
    import tiktoken
    enc = tiktoken.encoding_for_model('text-embedding-ada-002')
    total_tokens = sum([len(enc.encode(page.page_content)) for page in texts])
    print(f'Total tokens : {total_tokens}')
    print(f'Embedding cost in USD: {total_tokens / 1000 * 0.0004:.6f}')

# print_embedding_cost(chunks)

def insert_of_fetch_embeddings(collection_name):
    import chromadb
    # from chromadb.config import Settings
    from langchain.vectorstores.chroma import Chroma
    from langchain.embeddings.openai import OpenAIEmbeddings

    embeddings = OpenAIEmbeddings()
    client = chromadb.PersistentClient(path="./vector_db")
    collection = client.get_or_create_collection(collection_name)
    vector_store = Chroma(
        client=client,
        collection_name=collection_name,
        embedding_function=embeddings,
    )
    # collection.id
    print(collection)
        
    return vector_store 


def ask_and_get_answer(vector_store, q):
    from langchain.chains import RetrievalQA
    from langchain.chat_models import ChatOpenAI

    llm = ChatOpenAI(model='gpt-3.5-turbo', temperature=1)
    retrival = vector_store.as_retriever(search_type='similarity', search_kwargs={'k': 3})
    chain = RetrievalQA.from_chain_type(llm=llm, chain_type='stuff', retriever=retrival )
    answer = chain.run(q)
    return answer

def ask_with_memory(vector_store, q, chat_history=[]):
    from langchain.chains import ConversationalRetrievalChain
    from langchain.chat_models import ChatOpenAI

    llm = ChatOpenAI(model='gpt-3.5-turbo', temperature=1)
    retrival = vector_store.as_retriever(search_type='similarity', search_kwargs={'k': 3})
    crc = ConversationalRetrievalChain.from_llm(llm=llm, chain_type='stuff', retriever=retrival )
    result = crc({"question": q, "chat_history": chat_history})
    chat_history.append((q, result["answer"]))
    # answer = chain.run(q)
    return result, chat_history


# load data -––-----
data = load_pdf('pdfs/RefundCancellationRules_231209_173814.pdf')
chunks = chunk_data(data)
vs = insert_of_fetch_embeddings('irctcRefundRule')
chat_history = []
while True:
    q = input('Question : ')
    if (q.lower() in ['exit', 'quit']): 
        print('bye')
        break
    result, chat_history = ask_with_memory(vs, q, chat_history)
    print(f'Answer : {result["answer"]}')
    print('--' * 50)