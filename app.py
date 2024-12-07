from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
import streamlit as st
import os
import dotenv

dotenv.load_dotenv()

namespace = "stock-descriptions2"

sectors = [
    "Any Sector",
    "Consumer Defensive",
    "Communication Services",
    "Real Estate",
    "Healthcare",
    "Basic Materials",
    "Utilities",
    "Energy",
    "Industrials",
    "Consumer Cyclical",
    "Financial Services",
    "Technology",
]
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
pinecone_index = pc.Index("stocks")


st.set_page_config(page_title="Stock Finder", page_icon="📈")

st.title("Stock Finder")

query = st.text_input(
    "What kind of stock are you looking for?", "Cryptocurrency stocks"
)
embedded_query = SentenceTransformer("sentence-transformers/all-mpnet-base-v2").encode(
    query
)

# max market_cap for apple: 3670720643072
market_cap_filter = st.slider(
    "Market Cap Filter (in millions)",
    0,
    int(3670720643072 // 1e6),
    (0, int(3670720643072 // 1e6)),
)

# max volume for HMBL: 647676134
volume_filter = st.slider(
    "Volume Filter (in thousands)", 0, int(647676134 // 1e3), (0, int(647676134 // 1e3))
)

# sector
sector_filter = st.selectbox("Sector", sectors)
if sector_filter == "Any Sector":
    sector_filter = None
st.write("market_cap_filter:", market_cap_filter)
st.write("market_cap_min:", market_cap_filter[0])

searching = st.button("Search", use_container_width=True)
if searching:
    # query
    st.write("Searchin")

    embedded_query = embedded_query.tolist()

    result = None
    st.write("sector_filter:", sector_filter)
    if sector_filter:
        st.write("searching with sector filter")
        result = pinecone_index.query(
            query_vector=embedded_query,
            top_k=10,
            filter={
                # "Market Cap": {"$gte": market_cap_filter[0] * 1e6, "$lte": market_cap_filter[1] * 1e6},
                "Market Cap": {"$gte": 1082203647, "$lte": 1082203649},
                "Volume": {"$gte": volume_filter[0] * 1e3, "$lte": volume_filter[1] * 1e3},
               # "Sector": {"$eq": sector_filter},
            },
            include_metadata=True,
        )
    else:
        st.write("searching with no sector filter")
        result = pinecone_index.query(
            namespace=namespace,
            vector=embedded_query,
            top_k=10,
            filter={
                "Market Cap": {"$gte": market_cap_filter[0] * 1e6, "$lte": market_cap_filter[1] * 1e6},
                "Volume": {"$gte": volume_filter[0] * 1e3, "$lte": volume_filter[1] * 1e3},
            },
            include_metadata=True,
        )

        print('result', result)

