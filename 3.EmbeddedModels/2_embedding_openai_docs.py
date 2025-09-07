from langchain_openai import OpenAIEmbeddings

from dotenv import load_dotenv

load_dotenv()

embedding = OpenAIEmbeddings(model='text-embedding-3-large', dimensions=32) # more dimension more contextual meaning

documents = [
    "Delhi is the capital of India",
    "Kolakata is the capital of West Bengal",
    "Paris is the Capital of France"
]

result = embedding.embed_documents(documents) #convert my text into vector of 32 dimensions

print(str(result))

