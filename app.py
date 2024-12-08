from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
import streamlit as st
import os
import dotenv

dotenv.load_dotenv()

namespace = "stock-descriptions2"

sectors = [
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

def truncate_text(text, max_length=135):
    return text[:max_length] + '...' if len(text) > max_length else text

st.set_page_config(page_title="Stock Finder", page_icon="📈")

st.title("Stock Finder 📈")

query = st.text_input(
    "What kind of stock are you looking for?", "Cryptocurrency stocks"
)
embedded_query = SentenceTransformer("sentence-transformers/all-mpnet-base-v2").encode(
    query
)
embedded_query = embedded_query.tolist()

col1, col2 = st.columns(2)

with col1:
    number_filter = st.number_input(
    "Number of results",
    1,
    25,
    10,
    help="Specify the number of stock results to display."
    )
   
    sector_filter = st.selectbox(
        "Sector",
        ["Any Sector"] + sectors,
        help="Select a specific sector or 'Any Sector' for all sectors."
    )
    if sector_filter == "Any Sector":
        sector_filter = None

with col2:
    market_cap_filter = st.slider(
        "Market Cap Filter (in billions USD)",
        0.0,
        3670.0,  # Max market cap for Apple in billions
        (0.0, 3670.0),
        format="$%.1fB",
        help="Filter stocks by market capitalization range."
    )
    
    volume_filter = st.slider(
        "Volume Filter (in millions)",
        0,
        648,  # Max volume for HMBL in millions
        (0, 648),
        format="%d M",
        help="Filter stocks by trading volume range."
    )

searching = st.button("Search", use_container_width=True)
if searching:
    result = None
    with st.spinner('Searching...'):
        if sector_filter:
            result = pinecone_index.query(
                namespace=namespace,
                vector=embedded_query,
                top_k=number_filter,
                filter={
                    "Market Cap": {"$gte": market_cap_filter[0] * 1e9, "$lte": market_cap_filter[1] * 1e9},
                    "Volume": {"$gte": volume_filter[0] * 1e6, "$lte": volume_filter[1] * 1e6},
                    "Sector": {"$eq": sector_filter,},
                    "Ticker": {"$ne": "N/A"},
                },
                include_metadata=True,
            )
        else:
            result = pinecone_index.query(
                namespace=namespace,
                vector=embedded_query,
                top_k=number_filter,
                filter={
                    "Market Cap": {"$gte": market_cap_filter[0] * 1e9, "$lte": market_cap_filter[1] * 1e9},
                    "Volume": {"$gte": volume_filter[0] * 1e6, "$lte": volume_filter[1] * 1e6},
                    "Ticker": {"$ne": "N/A"},
                },
                include_metadata=True,
            )

    
    cols = st.columns(2)

    if len(result.matches) == 0:
        st.warning("No results found. Please try a different search.")
    
    for idx, obj in enumerate(result.matches):
        col_idx = idx % 2
        with cols[col_idx]:
            # Get metadata for display
            metadata = obj.metadata
            name = truncate_text(metadata.get('Name', 'N/A'), max_length=25)

            ticker = metadata.get('Ticker', 'N/A')

            description = truncate_text(metadata.get('text', 'No description available.'))

            sector = metadata.get('Sector', 'Sector Unknown')

            earnings_growth = metadata.get('Earnings_Growth', 0.0)
            earnings_growth = f"{earnings_growth*100:.2f}%"

            revenue_growth = metadata.get('Revenue_Growth', 'N/A')
            revenue_growth = f"{revenue_growth*100:.2f}%"

            gross_margins = metadata.get('Gross_Margins', 'N/A')
            gross_margins = f"{gross_margins*100:.2f}%"

            ebidta_margins = metadata.get('EBIDTA_Margins', 'N/A')
            ebidta_margins = f"{ebidta_margins*100:.2f}%"

            week_change = metadata.get('52_Week_Change', 'N/A')
            week_change = f"{week_change*100:.2f}%"

            website = metadata.get('website', 'www.finviz.com')
            if website == 'N/A':
                website = f"https://www.finviz.com/quote.ashx?t={ticker}"

            beta = metadata.get('Beta', 0.0)
            beta = f"{beta:.2f}"
            
            # Display the information in a styled box with similar design
            with st.expander(f"{ticker}", expanded=True):
                st.markdown(f"""
                <style>
                    .stock-card {{
                        background-color: #121212;
                        color: #ffffff;
                        padding: 20px;
                        border-radius: 8px;
                        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
                        max-width: 400px;
                    }}
                    .stock-name {{
                        font-size: 24px;
                        font-weight: 600;
                        margin-bottom: 4px;
                    }}
                    .stock-ticker {{
                        font-size: 16px;
                        color: #888888;
                        margin-bottom: 8px;
                    }}
                    .stock-sector {{
                        font-size: 14px;
                        color: #888888;
                        margin-bottom: 12px;
                    }}
                    .stock-description {{
                        color: #888888;
                        font-size: 14px;
                        line-height: 1.4;
                        margin-bottom: 12px;
                    }}
                    .stock-link {{
                        color: #3ea6ff;
                        text-decoration: none;
                        font-size: 14px;
                        display: block;
                        margin-bottom: 20px;
                    }}
                    .metrics-grid {{
                        display: grid;
                        grid-template-columns: 1fr 1fr;
                        gap: 16px;
                        margin-bottom: 16px;
                    }}
                    .metric-item {{
                        display: flex;
                        flex-direction: column;
                    }}
                    .metric-label {{
                        color: #888888;
                        font-size: 12px;
                        margin-bottom: 4px;
                    }}
                    .metric-value {{
                        font-size: 16px;
                        font-weight: 500;
                    }}
                    .metric-value.positive {{
                        color: #4caf50;
                    }}
                    .metric-value.negative {{
                        color: #f44336;
                    }}
                </style>

                <div class="stock-card">
                    <div class="stock-name">{ticker}</div>
                    <div class="stock-ticker">{name}</div>
                    <div class="stock-sector">{sector}</div>
                    <div class="stock-description">{description}</div>
                    <a href="{website}" class="stock-link" target="_blank">{website}</a>
                    <div class="metrics-grid">
                        <div class="metric-item">
                            <div class="metric-label">Earnings Growth</div>
                            {f'''<div class="metric-value positive">{earnings_growth}</div> ''' if float(earnings_growth.strip('%')) >= 0 else f'''<div class="metric-value negative">{earnings_growth}</div>'''}
                        </div>
                        <div class="metric-item">
                            <div class="metric-label">Revenue Growth</div>
                            {f'''<div class="metric-value positive">{revenue_growth}</div> ''' if float(revenue_growth.strip('%')) >= 0 else f'''<div class="metric-value negative">{revenue_growth}</div>'''}
                        </div>
                        <div class="metric-item">
                            <div class="metric-label">Gross Margins</div>
                            <div class="metric-value">{gross_margins}</div>
                        </div>
                        <div class="metric-item">
                            <div class="metric-label">EBITDA Margins</div>
                            <div class="metric-value">{ebidta_margins}</div>
                        </div>
                        <div class="metric-item">
                            <div class="metric-label">52 Week Change</div>
                            {f'''<div class="metric-value positive">{week_change}</div> ''' if float(week_change.strip('%')) >= 0 else f'''<div class="metric-value negative">{week_change}</div>'''}
                        </div>
                        <div class="metric-item">
                            <div class="metric-label">Beta</div>
                            <div class="metric-value">{beta}</div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)