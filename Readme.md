viral_shorts_factory/
│
├── .env
├── requirements.txt
├── main.py                     <-- (The code you just pasted)
│
├── data/
│   ├── inputs/                 <-- (Where local .mp4s go / where YouTube downloads to)
│   ├── renders/                <-- (Where your final Shorts will appear)
│   └── chroma_db/              <-- (Where your vector database lives)
│
└── src/
    ├── core/
    │   └── state_manager.py    <-- (The SQLite Queue)
    ├── editing/
    │   ├── ffmpeg_engine.py    <-- (The video/audio rendering layer)
    │   └── whisper_engine.py   <-- (The subtitle generator)
    ├── memory/
    │   └── chroma_client.py    <-- (The Vector DB logic)
    └── utils/
        ├── llm_client.py       <-- (The Colab Fallback wrapper)
        └── youtube_downloader.py<-- (The yt-dlp fetching tool)