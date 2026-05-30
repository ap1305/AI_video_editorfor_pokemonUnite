import os
import chromadb
from openai import OpenAI
from dotenv import load_dotenv

# Load your OpenAI key from the .env file
load_dotenv()
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def get_openai_embedding(text: str):
    """Uses OpenAI's newest, cheapest text model to understand meme slang."""
    response = openai_client.embeddings.create(
        input=text,
        model="text-embedding-3-small"
    )
    return response.data[0].embedding

def index_meme_library():
    print("\n" + "="*40)
    print("🧠 BOOTING OPENAI MEME INDEXER")
    print("="*40)

    meme_dir = os.path.join("assets", "memes")
    
    if not os.path.exists(meme_dir):
        os.makedirs(meme_dir)
        print(f"🚨 Folder '{meme_dir}' did not exist. I just created it.")
        return
        
    meme_files = [f for f in os.listdir(meme_dir) if f.endswith(('.mp4', '.mov', '.webm', '.gif', '.png', '.jpg'))]
    
    if not meme_files:
        print(f"🚨 No files found in '{meme_dir}'.")
        return

    # Connect to ChromaDB
    db_path = os.path.join("data", "meme_db")
    chroma_client = chromadb.PersistentClient(path=db_path)
    
    # We create the collection. If it exists from our SigLIP test, we should reset it 
    # so the 768d math doesn't clash with OpenAI's 1536d math.
    try:
        chroma_client.delete_collection(name="meme_index")
    except Exception:
        pass # Collection didn't exist yet
        
    meme_index = chroma_client.create_collection(name="meme_index")

    print(f"\n📂 Found {len(meme_files)} memes. Sending to OpenAI for slang analysis...")

    for file_name in meme_files:
        clean_text = os.path.splitext(file_name)[0].replace("_", " ").replace("-", " ")
        
        # Get the 1536-dimensional math from OpenAI
        vector = get_openai_embedding(clean_text)

        meme_index.add(
            embeddings=[vector],
            ids=[file_name], 
            metadatas=[{"file_path": os.path.join(meme_dir, file_name), "description": clean_text}]
        )
        print(f"✅ Indexed: {file_name} -> Understood as: '{clean_text}'")

    print("\n🎉 Meme Library fully indexed via OpenAI!")

if __name__ == "__main__":
    index_meme_library()