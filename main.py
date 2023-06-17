import os
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
import pinecone

pinecone.init(api_key=os.environ["PINECONE_API_KEY"], environment="asia-southeast1-gcp-free")

if __name__ == "__main__":
    print("Hello, Pinecone!")
    loader = TextLoader("./mediumblogs/mediumblog1.txt")
    document = loader.load()

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.split_documents(document)



    embeddings = OpenAIEmbeddings(openai_api_key=os.environ["OPENAI_API_KEY"])

    docsearch = Pinecone.from_documents(texts, embeddings, index_name="medium-blogs-embeddings-index")

    
    print(len(texts))