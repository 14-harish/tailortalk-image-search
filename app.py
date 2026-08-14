import io

import requests
import streamlit as st
from PIL import Image

from fashion_search import search_image
from agent import run_agent


# ==========================================
# Page configuration
# ==========================================

st.set_page_config(
    page_title="Saree Similarity Agent",
    page_icon="🥻",
    layout="wide"
)


# ==========================================
# Title
# ==========================================

st.title("🥻 Saree Similarity Agent")

st.write(
    "Upload a saree image or provide an image URL "
    "to find visually similar sarees from the catalogue."
)


# ==========================================
# Inputs
# ==========================================

uploaded_file = st.file_uploader(
    "Upload a saree image",
    type=["jpg", "jpeg", "png", "webp"]
)

st.write("**OR**")

image_url = st.text_input(
    "Enter an image URL"
)


# ==========================================
# Search
# ==========================================

if st.button(
    "🔎 Find Similar Sarees",
    type="primary"
):

    image = None
    results = []

    # ==========================================
    # Uploaded image
    # ==========================================

    if uploaded_file is not None:

        try:

            image = Image.open(
                uploaded_file
            ).convert("RGB")

        except Exception as e:

            st.error(
                f"Could not open uploaded image: {e}"
            )


    # ==========================================
    # Image URL
    # ==========================================

    elif image_url:

        try:

            response = requests.get(
                image_url,
                timeout=30
            )

            response.raise_for_status()

            image = Image.open(
                io.BytesIO(
                    response.content
                )
            ).convert("RGB")

        except Exception as e:

            st.error(
                f"Could not load image URL: {e}"
            )

    else:

        st.warning(
            "Please upload an image or enter an image URL."
        )


    # ==========================================
    # Search catalogue
    # ==========================================

    if image is not None:

        st.subheader(
            "Query Image"
        )

        st.image(
            image,
            width=300
        )

        with st.spinner(
            "Finding visually similar sarees..."
        ):

            try:

                # ----------------------------------
                # URL input:
                # Use LLM tool-calling agent
                # ----------------------------------

                if image_url:

                    agent_output = run_agent(
                        image_url
                    )

                    tool_result = agent_output.get(
                        "tool_result"
                    )

                    if (
                        isinstance(tool_result, dict)
                        and "results" in tool_result
                    ):

                        results = tool_result["results"]

                    elif (
                        isinstance(tool_result, dict)
                        and "error" in tool_result
                    ):

                        st.error(
                            tool_result["error"]
                        )

                    else:

                        st.error(
                            "The agent did not return search results."
                        )


                # ----------------------------------
                # Uploaded image:
                # Direct FashionCLIP search
                # ----------------------------------

                else:

                    results = search_image(
                        image,
                        top_k=5
                    )

            except Exception as e:

                st.error(
                    f"Search failed: {e}"
                )

                results = []


        # ======================================
        # Results
        # ======================================

        if results:

            st.subheader(
                "Similar Sarees"
            )

            for rank, result in enumerate(
                results,
                start=1
            ):

                st.markdown(
                    f"### {rank}. "
                    f"{result['name']}"
                )

                col1, col2 = st.columns(
                    [1, 2]
                )

                with col1:

                    st.image(
                        result["image_url"],
                        use_container_width=True
                    )

                with col2:

                    st.write(
                        f"**SKU:** "
                        f"{result['sku']}"
                    )

                    st.write(
                        f"**Similarity:** "
                        f"{result['score']:.4f}"
                    )

                    st.write(
                        f"**Retail Price:** "
                        f"₹{result['retail_price']:,}"
                    )

                    st.write(
                        f"**Discounted Price:** "
                        f"₹{result['discounted_price']:,}"
                    )

                    st.write(
                        f"**Stock:** "
                        f"{result['stock']}"
                    )

                    st.link_button(
                        "View Product",
                        result["website_link"]
                    )

                st.divider()

        elif image is not None:

            st.warning(
                "No similar sarees were found."
            )