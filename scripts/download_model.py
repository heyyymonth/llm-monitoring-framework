"""
Model Pre-downloading Script

This script is executed during the Docker build process to pre-download the 
necessary sentence-transformer models. By doing this, we "bake" the model into
the Docker image, which significantly speeds up the application's startup time
in a containerized environment.
"""
from sentence_transformers import SentenceTransformer

def download_model():
    """
    Downloads and caches the specified sentence-transformer model.
    """
    model_name = 'all-MiniLM-L6-v2'
    print(f"Downloading and caching model: {model_name}")
    try:
        SentenceTransformer(model_name)
        print("Model downloaded and cached successfully.")
    except Exception as e:
        print(f"Error downloading model: {e}")
        # Exit with a non-zero status code to fail the Docker build if download fails
        exit(1)

if __name__ == "__main__":
    download_model() 