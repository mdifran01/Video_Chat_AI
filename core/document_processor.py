from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

def segment_to_doc(all_segments :list) ->list:
    documents=[]
    
    for segment in all_segments:
        document=Document(
            page_content=segment["text"],
            
            metadata={
                "source": segment["source"],
                "chunk_id":segment["chunk_id"],
                "start":segment["start"],
                "end":segment["end"]
                
                
            }
        )
        documents.append(document)
        
    return documents    


def split_document(documents:list)-> list:
    text_splitter=RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150)
    
    chunks= text_splitter.split_documents(documents)
    
    return chunks

    