from langchain.document_loaders import PyPDFLoader

# Load PDF
loader = PyPDFLoader("lecture_notes.pdf")
pdf_docs = loader.load()

print(f"Total pages: {len(pdf_docs)}")
print("First page content:\n", pdf_docs[0].page_content[:500])
print("Metadata:", pdf_docs[0].metadata)
