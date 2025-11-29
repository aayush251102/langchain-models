from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader('submit.pdf')

docs = loader.load()

print(len(docs))