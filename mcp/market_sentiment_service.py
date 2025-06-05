from mcp.server.fastmcp import FastMCP
import requests
from bs4 import BeautifulSoup
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from typing import Dict, Any, List
from datetime import datetime
from serpapi import GoogleSearch

# 检查并下载（如果不存在）
try:
    nltk.data.find('sentiment/vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')
mcp = FastMCP()

class MarketSentimentService(FastMCP):
    def __init__(self):
        super().__init__()
        self.sia = SentimentIntensityAnalyzer()
        self.serpapi_key = "14fa87fa2f0698c7f24b7cd7311c5f4e889c4f2df465a07ef366145c24ae4b74"
        
    @mcp.tool()
    async def web_browsing(self, query: str) -> Dict[str, Any]:
        """
        Collect market information for a given stock or financial topic using SerpApi.
        
        Args:
            query: The stock symbol or financial topic to analyze
            
        Returns:
            Dict containing collected market information
        """
        try:
            # Prepare search query
            search_query = f"{query} financial news"
            
            # Set up SerpApi parameters
            params = {
                "q": search_query,
                "hl": "en",
                "gl": "us",
                "api_key": self.serpapi_key,
                "num": 5  # Get top 5 results
            }

            # Perform search
            search = GoogleSearch(params)
            result = search.get_dict()
            
            # Extract news data
            news_data = []
            for item in result.get("organic_results", [])[:5]:
                news_data.append({
                    'title': item.get('title', 'No title available'),
                    'url': item.get('link', ''),
                    'publish_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'source': item.get('source', 'Unknown'),
                    'snippet': item.get('snippet', '')
                })

            # If no news data was collected, add a default message
            if not news_data:
                news_data.append({
                    'title': f'No recent news available for {query}',
                    'source': 'System',
                    'publish_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'snippet': ''
                })

            return {
                'type': 'topic',
                'topic': query,
                'news_data': news_data
            }
            
        except Exception as e:
            print(f"Error in web_browsing: {str(e)}")
            return {
                'error': str(e),
                'news_data': [{
                    'title': f'Error occurred while fetching data: {str(e)}',
                    'source': 'System',
                    'publish_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'snippet': ''
                }]
            }

    @mcp.tool()
    async def sentiment_analysis(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze sentiment and generate a comprehensive report for the collected market data.
        
        Args:
            market_data: Dictionary containing market information from web_browsing
            
        Returns:
            Dict containing sentiment analysis report
        """
        try:
            if 'error' in market_data:
                return {'error': market_data['error']}

            news_data = market_data.get('news_data', [])
            if not news_data:
                return {'error': 'No news data available for analysis'}

            # Analyze sentiment for each news item
            sentiment_results = []
            for news in news_data:
                sentiment = self.sia.polarity_scores(news['title'])
                sentiment_results.append({
                    'title': news['title'],
                    'sentiment_scores': {
                        'compound': sentiment['compound'],
                        'positive': sentiment['pos'],
                        'negative': sentiment['neg'],
                        'neutral': sentiment['neu']
                    },
                    'source': news.get('source', 'Unknown'),
                    'publish_time': news.get('publish_time', 'Unknown')
                })

            # Calculate overall sentiment metrics
            compound_scores = [r['sentiment_scores']['compound'] for r in sentiment_results]
            avg_sentiment = sum(compound_scores) / len(compound_scores)
            
            # Generate sentiment report
            report = {
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'overall_sentiment': {
                    'average_score': avg_sentiment,
                    'sentiment_level': self._get_sentiment_level(avg_sentiment),
                    'confidence': self._calculate_confidence(sentiment_results)
                },
                'detailed_analysis': sentiment_results,
                'market_context': {}
            }

            # Add market context if available
            if market_data['type'] == 'stock':
                report['market_context'] = {
                    'stock_info': market_data['stock_info'],
                    'price_change': self._calculate_price_change(market_data['stock_info'])
                }
            else:
                report['market_context'] = {
                    'topic': market_data['topic']
                }

            return report

        except Exception as e:
            return {'error': str(e)}

    def _get_sentiment_level(self, score: float) -> str:
        """Convert sentiment score to descriptive level"""
        if score >= 0.5:
            return "Very Positive"
        elif score >= 0.1:
            return "Positive"
        elif score > -0.1:
            return "Neutral"
        elif score > -0.5:
            return "Negative"
        else:
            return "Very Negative"

    def _calculate_confidence(self, sentiment_results: List[Dict]) -> float:
        """Calculate confidence level based on sentiment consistency"""
        if not sentiment_results:
            return 0.0
        
        # Calculate standard deviation of sentiment scores
        scores = [r['sentiment_scores']['compound'] for r in sentiment_results]
        mean = sum(scores) / len(scores)
        variance = sum((x - mean) ** 2 for x in scores) / len(scores)
        std_dev = variance ** 0.5
        
        # Convert to confidence score (0-1)
        # Lower standard deviation means higher confidence
        confidence = 1 - min(std_dev, 1)
        return round(confidence, 2)

    def _calculate_price_change(self, stock_info: Dict) -> Dict:
        """Calculate price change metrics"""
        current = stock_info.get('current_price', 0)
        previous = stock_info.get('previous_close', 0)
        if previous == 0:
            return {'percentage': 0, 'direction': 'unchanged'}
        
        change = ((current - previous) / previous) * 100
        return {
            'percentage': round(change, 2),
            'direction': 'up' if change > 0 else 'down' if change < 0 else 'unchanged'
        }

if __name__ == "__main__":
    service = MarketSentimentService()
    service.run() 