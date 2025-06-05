# Market Sentiment Analysis MCP Service

This MCP service provides market sentiment analysis for stocks and financial topics using two separate tools: web browsing and sentiment analysis.

## Features

- Stock information collection using Yahoo Finance data
- General financial topic information collection using web search
- Comprehensive sentiment analysis using NLTK's VADER sentiment analyzer
- Detailed sentiment reports with confidence scores
- Market context analysis including price changes and trends

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Run the service:
```bash
python market_sentiment_service.py
```

## Usage

The service provides two tools that work together:

1. `web_browsing` tool:
   - Input: Stock symbol (e.g., "AAPL", "GOOGL") or financial topic (e.g., "cryptocurrency")
   - Returns: Raw market data including news articles and market information
   - For stocks: Returns stock details and recent news
   - For topics: Returns relevant news headlines

2. `sentiment_analysis` tool:
   - Input: Market data from web_browsing tool
   - Returns: Comprehensive sentiment analysis report including:
     - Overall sentiment score and level
     - Confidence score
     - Detailed analysis of each news item
     - Market context (price changes for stocks)

## Example Usage Flow

1. First, collect market data:
```python
market_data = await service.web_browsing("AAPL")
```

2. Then, analyze the sentiment:
```python
sentiment_report = await service.sentiment_analysis(market_data)
```

## Response Format

The sentiment analysis report includes:
- Timestamp of analysis
- Overall sentiment metrics:
  - Average sentiment score (-1 to 1)
  - Sentiment level (Very Positive to Very Negative)
  - Confidence score (0 to 1)
- Detailed analysis of each news item
- Market context:
  - For stocks: Stock info and price changes
  - For topics: Topic name and related metrics

## Example Prompts

- "Analyze the market sentiment for AAPL"
- "What's the current sentiment about cryptocurrency?"
- "How is the market feeling about Tesla (TSLA)?"

## Note

- Sentiment scores range from -1 (most negative) to 1 (most positive)
- Confidence scores range from 0 (low confidence) to 1 (high confidence)
- The service uses NLTK's VADER sentiment analyzer for accurate sentiment analysis 