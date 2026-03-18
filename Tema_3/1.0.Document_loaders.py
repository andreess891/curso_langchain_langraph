from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader

loader = PyPDFLoader("Cv_Jonatan.pdf")

pages = loader.load()

#print(dir(pages[0]))

for i, page in enumerate(pages):
    print(f"=== Pagina {i+1} ===")
    print(f"Contenido: {page.page_content}")
    print(f"Metadatos: {page.metadata}")

print("\n\n")
print("=== Cargando contenido web ===")

loader2 = WebBaseLoader("https://techmind.ac/")

web = loader2.load()

print(web)