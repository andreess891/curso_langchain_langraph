from langchain_community.document_loaders import GoogleDriveLoader

credentials_path = "credentials.json"
token_path = "token.json"

loader = GoogleDriveLoader(
    folder_id="1Jh4fLdU_Br7991hdxmttAr7nhOijYe9U",
    credentials_path=credentials_path,
    token_path=token_path,
    recursive=True
)

documents = loader.load()

#print(documents)

print(f"Metadatos del primer documento: {documents[0].metadata}")
print(f"Contenido del primer documento: {documents[0].page_content}")
