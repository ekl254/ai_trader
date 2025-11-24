"""Test NewsAPI and Massive.com integrations."""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

print("🧪 Testing API Integrations\n")
print("=" * 60)

# Test 1: NewsAPI
print("\n1. Testing NewsAPI...")
try:
    from src.newsapi_client import newsapi_client
    
    articles = newsapi_client.get_stock_news("AAPL", days_back=1, max_articles=5)
    print(f"   ✅ NewsAPI Working - Fetched {len(articles)} articles for AAPL")
    if articles:
        print(f"   📰 Latest: {articles[0].get('title', 'N/A')[:60]}...")
except Exception as e:
    print(f"   ❌ NewsAPI Error: {e}")

# Test 2: Massive.com API
print("\n2. Testing Massive.com API...")
try:
    from src.massive_client import massive_client
    
    snapshot = massive_client.get_snapshot("AAPL")
    if snapshot:
        print(f"   ✅ Massive.com Working - Got snapshot for AAPL")
        print(f"   📊 Day High/Low: ${snapshot.get('day', {}).get('h', 'N/A')} / ${snapshot.get('day', {}).get('l', 'N/A')}")
    else:
        print("   ⚠️  Snapshot returned None (may be rate limited or market closed)")
except Exception as e:
    print(f"   ❌ Massive API Error: {e}")

# Test 3: Sentiment Analysis
print("\n3. Testing Sentiment Analysis with NewsAPI...")
try:
    from src.sentiment import get_news_sentiment
    
    sentiment_score = get_news_sentiment("AAPL", max_articles=10)
    print(f"   ✅ Sentiment Analysis Working")
    print(f"   📈 AAPL Sentiment Score: {sentiment_score:.2f}/100")
    
    if sentiment_score > 60:
        print("   🟢 Positive sentiment")
    elif sentiment_score < 40:
        print("   🔴 Negative sentiment")
    else:
        print("   🟡 Neutral sentiment")
except Exception as e:
    print(f"   ❌ Sentiment Error: {e}")

# Test 4: Environment Variables
print("\n4. Checking Environment Variables...")
env_vars = {
    "NEWSAPI_KEY": os.getenv("NEWSAPI_KEY"),
    "MASSIVE_API_KEY": os.getenv("MASSIVE_API_KEY"),
    "ALPACA_API_KEY": os.getenv("ALPACA_API_KEY"),
}

for key, value in env_vars.items():
    if value:
        masked = value[:8] + "..." if len(value) > 8 else value
        print(f"   ✅ {key}: {masked}")
    else:
        print(f"   ❌ {key}: Not set")

print("\n" + "=" * 60)
print("\n✨ Integration Test Complete!\n")
