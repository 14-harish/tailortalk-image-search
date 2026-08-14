import io
import requests

from PIL import Image
from langchain_core.tools import tool

from fashion_search import search_image


@tool
def search_sarees(image_url: str) -> dict:
    """
    Find sarees visually similar to an image URL.

    Use this tool when the user asks to find,
    search for, or recommend sarees visually
    similar to a given image.
    """

    if not image_url:
        return {
            "error": "No image URL was provided."
        }

    try:
        response = requests.get(
            image_url,
            timeout=30
        )

        response.raise_for_status()

        image = Image.open(
            io.BytesIO(response.content)
        ).convert("RGB")

    except Exception as e:
        return {
            "error": f"Could not load image: {e}"
        }

    try:
        results = search_image(
            image,
            top_k=5
        )

        # Format scores for cleaner agent output
        for result in results:
            result["score"] = round(
                result["score"],
                4
            )

        return {
            "results": results
        }

    except Exception as e:
        return {
            "error": f"Similarity search failed: {e}"
        }