from openai import OpenAI
from market_sentiment_service import MarketSentimentService
import json
from typing import Dict, Any, List

class MarketChatAssistant:
    def __init__(self):
        self.client = OpenAI(
            api_key="sk-9ccaa6edf3a54a8895fa6d3520c7f8fe"
        )
        self.market_service = MarketSentimentService()
        
        # Define available functions
        self.available_functions = {
            "web_browsing": self.market_service.web_browsing,
            "sentiment_analysis": self.market_service.sentiment_analysis
        }
        
        # Define function descriptions for the model
        self.function_descriptions = [
            {
                "name": "web_browsing",
                "description": "Collect market information for a given stock or financial topic",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "one company name(should not in symbol format likr AAPL)"
                        }
                    },
                    "required": ["query"]
                }
            },
            {
                "name": "sentiment_analysis",
                "description": "Analyze sentiment and generate a comprehensive report for the collected market data",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "market_data": {
                            "type": "object",
                            "description": "Dictionary containing market information from web_browsing"
                        }
                    },
                    "required": ["market_data"]
                }
            }
        ]

    async def process_message(self, user_message: str) -> str:
        """Process user message and generate response"""
        try:
            # Get the model's response with function calling
            client = OpenAI(
                api_key="sk-9ccaa6edf3a54a8895fa6d3520c7f8fe",
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            )
            completion = client.chat.completions.create(
                model="qwen-plus",
                messages=[
                    {"role": "system", "content": """You are a helpful market analysis assistant. 
                    You can help users analyze market sentiment for stocks and financial topics.
                    Use the available functions to gather and analyze market data.you must return one stock symbol in the response's function_call arguments"""},
                    {"role": "user", "content": user_message}
                ],
                functions=self.function_descriptions,
                function_call="auto",
                temperature=0.9,
                max_tokens=100,
                top_p=1,
                frequency_penalty=1,
                presence_penalty=0.5
            )

            # Get the assistant's message
            assistant_message = completion.choices[0].message
            print(assistant_message)
            print(assistant_message.function_call)
            
            # Check if the model wants to call a function
            if assistant_message.function_call:
                function_name = assistant_message.function_call.name
                function_args = json.loads(assistant_message.function_call.arguments)

                # Call the appropriate function
                if function_name in self.available_functions:
                    function_response = await self.available_functions[function_name](**function_args)
                    
                    # If this was web_browsing, automatically call sentiment_analysis
                    if function_name == "web_browsing":
                        # First format the web browsing results
                        web_browsing_summary = self._format_web_browsing_response(function_response)
                        
                        # Then get sentiment analysis
                        sentiment_response = await self.available_functions["sentiment_analysis"](
                            market_data=function_response
                        )
                        
                        # Combine both responses
                        return f"{web_browsing_summary}\n\n{self._format_response(sentiment_response)}"
                    else:
                        return self._format_response(function_response)
                else:
                    return "I apologize, but I couldn't process that request."

            return assistant_message.content

        except Exception as e:
            return f"An error occurred: {str(e)}"

    def _format_web_browsing_response(self, data: Dict[str, Any]) -> str:
        """Format the web browsing results into a user-friendly message"""
        if 'error' in data:
            return f"Error: {data['error']}"

        # Get news data
        news_data = data.get('news_data', [])
        
        # Create a summary of the news
        news_summary = []
        for item in news_data[:3]:  # Only use top 3 news items
            title = item.get('title', '')
            snippet = item.get('snippet', '')
            if title:
                news_summary.append(f"- {title}")
                if snippet:
                    # Add snippet with proper indentation
                    news_summary.append(f"  {snippet}")
        print(snippet+"??????????????????")
        # Format the response
        response = [
            f"Latest News for {data.get('topic', 'the topic')}:",
            *news_summary
        ]
        
        return "\n".join(response)

    def _format_response(self, data: Dict[str, Any]) -> str:
        """Format the function response into a user-friendly message"""
        if 'error' in data:
            return f"Error: {data['error']}"

        # Get overall sentiment
        sentiment = data.get('overall_sentiment', {})
        sentiment_level = sentiment.get('sentiment_level', 'Unknown')
        avg_score = sentiment.get('average_score', 0)
        
        # Get news data
        news_data = data.get('news_data', [])
        
        # Create a summary of the news
        news_summary = []
        for item in news_data[:3]:  # Only use top 3 news items
            title = item.get('title', '')
            if title:
                news_summary.append(f"- {title}")
        
        # Format the response
        response = [
            f"Market Sentiment Summary for {data.get('topic', 'the topic')}:",
            f"Overall Sentiment: {sentiment_level} (Score: {avg_score:.2f})",
            # "\nKey News Highlights:",
            # *news_summary
        ]
        
        return "\n".join(response)

async def main():
    assistant = MarketChatAssistant()
    
    print("Market Analysis Assistant (type 'quit' to exit)")
    print("Example queries:")
    print("- What's the market sentiment for AAPL?")
    print("- How is the market feeling about cryptocurrency?")
    print("- Analyze Tesla's market sentiment")
    
    while True:
        user_input = input("\nYour question: ")
        if user_input.lower() == 'quit':
            break
            
        response = await assistant.process_message(user_input)
        print("\nAssistant:", response)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main()) 