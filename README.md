# Search engine for short-form video content
The significance of this system lies in its transition from metadata-dependent search to deep content-aware retrieval. This system addresses diverse use-cases across both professional and consumer sectors. In the realm of digital asset management, content creators and marketing teams can leverage the platform to rapidly discover new or similar content, content reuse and perform analytics tasks based on the data offered. For general users, the system enables highly granular queries, such as locating a specific culinary technique or a niche DIY repair method mentioned in passing within a 30-second reel. Furthermore, the tool serves as an essential utility for academic researchers and fact-checkers who require the ability to conduct semantic searches across thousands of video hours to identify trends, verify context, or track the dissemination of information across platforms. Overall, this system bridges a major gap, helping take users from video consumption to personalized and customizable video retrieval. The main functionality offered by the system to the user is the ability to search video content using natural language, which is enabled by understanding their audio, visual content, and on-screen text.

> Read the docs for in detail descriptions, algorithms and flow diagrams: [https://github.com/1407arjun/video-rag-search/blob/main/docs/documentation.pdf](https://github.com/1407arjun/video-rag-search/blob/main/docs/documentation.pdf)

## System Overview
Python is used as the preferred programming language to develop this system. FFmpeg is used for all video related operations. The user interface is developed using Streamlit, and the system uses Qdrant as its vector storage and database. The following secrets are used, and have to be located at `.streamlit/secrets.toml` file at the root:

``` toml
[qdrant]
url = <qdrant_url>
api_key = <qdrant_api_key>

[llm]
endpoint = <hosted_openai_model_endpoint>
api_key = <openai_api_key>
```

---


This system consists of three main stages to serve as an end to end search engine: the search itself, discovery and ingestion (hereafter referred to as the video processing stage) and data management (hereafter referred to as the vectorization or vector operations stage). Each of the stages is briefly discussed below:

<img width="1815" height="808" alt="image" src="https://github.com/user-attachments/assets/8bddc2f3-25fe-41d0-b5f8-d365e9b101e7" />

### Search
The search functionality is the primary feature of this system. It contains a search bar to query the video library and also an interface which lists relevant videos according to their rank with respect to the search query.  The ranking mechanism uses the BM25 model to rank retrieved videos with respect to the search query. This is discussed in detail in section 4.3 of the documentation.

### Video Processing
The video processing functionality is essential to populate data in the vector database which is used for search. It accepts a video URL as an input, which is then downloaded for processing using the yt-dlp package. Once downloaded, the video is processed by an ingestion pipeline which contains a series of models (GPT 4.1 for visuals and Whisper for ASR) and utility functions (frame sampling) to capture as much information as possible from the video into text. The extracted text and frames are presented on the user interface to the user for review. The frame sampling and ingestion pipelines are discussed in detail in sections 4.1 and 4.2 of the documentation respectively.

### Vectorization
This stage is present on the user interface as the video library which allows users to view all ingested videos and their metadata that are stored in the vector database and also to delete them from the vector database. Beneath that, it also contains an insertion functionality which is used once a video is processed and reviewed by the user. It uses the text-embedding-3-small model by OpenAI to convert text into embeddings. The metadata is stored alongside its embedding in the vector database and fetched during retrieval. The insertion functionality is discussed in detail along with the video ingestion pipeline in section 4.2 of the documentation.

## Evaluations
We evaluated the system on returning the top-5 results for a query. We calculated the precision@k, recall@k, F1-score@k metrics for a data of 50 curated queries and 20 videos. The queries are a manually curated mixture of direct keyword matches and semantic matches corresponding to one or more videos. We tested our system against the existing search systems which are based on only captions and a system which uses captions and transcript together. Following are the results for K = 1, 3 and 5:

### Evaluation scores at K = 1

| Search with | precision@1 | recall@1 | F1-score@1 |
| :--- | :--- | :--- | :--- |
| Caption only (existing systems) | 0.6346 | 0.5247 | 0.5526 |
| Caption + transcript | 0.6923 | 0.5856 | 0.6135 |
| Current system | **0.9038** | **0.7535** | **0.7897** |

---

### Evaluation scores at K = 3

| Search with | precision@3 | recall@3 | F1-score@3 |
| :--- | :--- | :--- | :--- |
| Caption only (existing systems) | 0.6346 | 0.6119 | 0.6207 |
| Caption + transcript | 0.7179 | 0.6952 | 0.7040 |
| Current system | **0.9295** | **0.9026** | **0.9130** |

---

### Evaluation scores at K = 5

| Search with | precision@5 | recall@5 | F1-score@5 |
| :--- | :--- | :--- | :--- |
| Caption only (existing systems) | 0.6667 | 0.6667 | 0.6667 |
| Caption + transcript | 0.7247 | 0.7247 | 0.7247 |
| Current system | **0.9349** | **0.9349** | **0.9349** |

## License
This project is licensed under the [MIT License](LICENSE).


