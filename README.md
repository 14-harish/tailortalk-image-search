# TailorTalk Image Search

AI-powered visual search for sarees using FashionCLIP, Qdrant, Groq, and Streamlit.

The application allows users to upload a saree image or provide an image URL and retrieve visually similar sarees from a product catalogue.

## Live Application

https://tailortalk-image-search-harish.streamlit.app

## GitHub Repository

https://github.com/14-harish/tailortalk-image-search

---

## Features

- Upload a saree image for visual similarity search.
- Search using an image URL.
- Fashion-specific image embeddings using FashionCLIP.
- Vector similarity search using Qdrant.
- Product metadata retrieval from the catalogue.
- Displays product name, SKU, stock, prices, product image, and product link.
- Groq-based LLM agent with tool calling.
- Deployed on Streamlit Community Cloud.

---

## Architecture

```text
User Image / Image URL
          |
          v
    Streamlit App
          |
          v
      FashionCLIP
          |
          v
   512-D Embedding
          |
          v
       Qdrant
          |
          v
  Similarity Search
          |
          v
   Product SKU Lookup
          |
          v
   Catalogue Metadata
          |
          v
    Similar Sarees
```

For the agent workflow:

```text
User Request
     |
     v
    Groq
     |
     | tool call
     v
search_sarees
     |
     v
FashionCLIP + Qdrant
     |
     v
Search Results
     |
     v
    Groq
     |
     v
Final Response
```

---

## Technology Stack

### Frontend and Deployment

- Streamlit
- Streamlit Community Cloud

### Machine Learning

- FashionCLIP: `patrickjohncyh/fashion-clip`
- Hugging Face Transformers
- PyTorch

### Vector Database

- Qdrant

### LLM

- Groq

### Data Processing

- Python
- Pandas
- NumPy
- Pillow

---

## Model Choice

The initial implementation used OpenAI CLIP:

```text
openai/clip-vit-base-patch32
```

This provided a working baseline for visual similarity search, but the retrieved results were not consistently relevant to fashion-specific visual characteristics.

The implementation was then evaluated using FashionCLIP:

```text
patrickjohncyh/fashion-clip
```

FashionCLIP was selected for the final application because it is specifically trained for fashion-related image and text representations and produced more relevant retrieval results on the tested saree queries.

---

## Vector Database

Qdrant is used for nearest-neighbor vector similarity search.

The catalogue images were converted into normalized 512-dimensional FashionCLIP embeddings and stored in a Qdrant collection named:

```text
sarees_fashion
```

At query time:

1. The input image is converted into a FashionCLIP embedding.
2. The embedding is normalized.
3. Qdrant searches for the nearest catalogue embeddings.
4. The retrieved filenames are mapped to product SKUs.
5. Product metadata is retrieved from the catalogue CSV.
6. The matching products are displayed to the user.

A pre-built Qdrant collection is included in the repository so the application does not need to regenerate catalogue embeddings during startup.

---

## Improving Search Quality

Several approaches were tested during development.

### 1. OpenAI CLIP baseline

The initial CLIP implementation provided general visual similarity but sometimes returned sarees that were visually different in important fashion-related attributes.

### 2. FashionCLIP

FashionCLIP was evaluated as a fashion-specific alternative.

The results were generally more relevant, particularly for queries where the catalogue contained visually related sarees.

The final deployed application therefore uses FashionCLIP for retrieval.

### 3. Colour-based reranking

A separate colour feature representation was also implemented and evaluated.

Two weighting strategies were tested:

```text
95% FashionCLIP + 5% Colour
85% FashionCLIP + 15% Colour
```

Colour reranking improved some queries but negatively affected others. Because the improvement was not consistent across the evaluation set, colour reranking was not used in the final deployed retrieval pipeline.

This allowed the final system to prioritize the stronger FashionCLIP representation instead of adding a feature that did not consistently improve retrieval quality.

---

## Dataset

The application uses a saree product catalogue containing:

- Product name
- SKU
- Stock
- Retail price
- Discounted price
- Image URL
- Website link

The catalogue metadata is stored in:

```text
byrappa_tejas_31july.csv
```

The original catalogue images were used to generate the embeddings. They are not required by the deployed application because the precomputed embeddings are stored in Qdrant and product images are loaded using their catalogue image URLs.

---

## Project Structure

```text
tailortalk-image-search/
│
├── app.py
├── agent.py
├── tools.py
├── fashion_search.py
│
├── build_color_features.py
├── build_fashion_qdrant.py
├── generate_fashion_embeddings.py
├── download_images.py
│
├── byrappa_tejas_31july.csv
├── requirements.txt
├── .env.example
├── .gitignore
│
└── data/
    └── qdrant/
        └── collection/
            └── sarees_fashion/
```

---

## Key Files

### `app.py`

Streamlit interface for:

- Image upload
- Image URL input
- Query image display
- Similarity search
- Product result display

### `fashion_search.py`

Core visual search implementation.

Responsible for:

- Loading FashionCLIP
- Generating image embeddings
- Loading Qdrant
- Performing similarity search
- Mapping results to catalogue metadata

### `tools.py`

Defines the `search_sarees` tool used by the Groq agent.

The tool accepts an image URL and returns visually similar sarees.

### `agent.py`

Implements the Groq-based agent workflow and tool calling.

### `build_fashion_qdrant.py`

Creates the Qdrant collection and stores the precomputed FashionCLIP embeddings.

### `generate_fashion_embeddings.py`

Generates FashionCLIP embeddings for the catalogue images.

### `build_color_features.py`

Generates colour features used during the colour-reranking experiments.

### `download_images.py`

Downloads catalogue images used during the embedding-generation process.

---

## Local Setup

Clone the repository:

```bash
git clone https://github.com/14-harish/tailortalk-image-search.git
cd tailortalk-image-search
```

Create a virtual environment:

```bash
python3 -m venv .venv
```

Activate it on macOS/Linux:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Create a `.env` file containing your Groq API key:

```env
GROQ_API_KEY=your_groq_api_key
```

Do not commit the `.env` file to GitHub.

Run the application:

```bash
streamlit run app.py
```

---

## Deployment

The application is deployed using Streamlit Community Cloud.

The deployment uses:

- The pre-built Qdrant collection included in the repository.
- FashionCLIP downloaded from Hugging Face during application initialization.
- Groq API credentials configured through Streamlit secrets.

No local setup is required for the reviewer.

Live application:

https://tailortalk-image-search-harish.streamlit.app

---

## Assumptions and Trade-offs

### Precomputed embeddings

Catalogue embeddings are precomputed rather than generated when the application starts.

This reduces startup time and avoids repeatedly processing the entire catalogue. The trade-off is that the embeddings must be regenerated when the catalogue changes.

### Local Qdrant storage

The project uses Qdrant's local storage mode.

This keeps the implementation simple and makes the project self-contained. For a larger production deployment with multiple application instances, a hosted Qdrant server would be more appropriate.

### CPU inference

The deployed application performs FashionCLIP inference on CPU.

This avoids requiring GPU infrastructure but results in higher inference latency compared with GPU inference.

### Catalogue metadata

Product metadata is stored in a CSV file and mapped to retrieved products using SKU information encoded in the image filenames.

A production system with frequently changing inventory would benefit from a dedicated database or product API.

### Similarity scores

Similarity scores represent embedding similarity and should not be interpreted as a percentage probability or a human-validated accuracy score.

---

## Future Improvements

Potential improvements include:

- Hosted Qdrant for larger-scale deployments.
- Metadata filtering by colour, price, stock, or saree type.
- Combining visual similarity with structured product metadata.
- Improved image preprocessing and background handling.
- Evaluation of additional fashion-specific embedding models.
- A larger labelled evaluation dataset with metrics such as Recall@K and NDCG.
- User feedback and relevance-based reranking.
- Caching model initialization and repeated searches.

---

## Environment Variables

The Groq agent requires:

```text
GROQ_API_KEY
```

See `.env.example` for the expected configuration.

The actual API key must be stored in the deployment platform's secrets and must not be committed to the repository.

---

## Links

Live Application:

https://tailortalk-image-search-harish.streamlit.app

GitHub Repository:

https://github.com/14-harish/tailortalk-image-search