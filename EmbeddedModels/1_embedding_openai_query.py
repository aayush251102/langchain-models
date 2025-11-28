from langchain_openai import OpenAIEmbeddings

from dotenv import load_dotenv

load_dotenv()

embedding = OpenAIEmbeddings(model='text-embedding-3-large', dimensions=32) # more dimension more contextual meaning

result = embedding.embed_query("I am Aayush Kashyap.") #convert my text into vector of 32 dimensions

print(str(result))

