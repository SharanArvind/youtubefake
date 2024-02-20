import os
import json
from googleapiclient.discovery import build
from transformers import TFAutoModelForCausalLM, AutoTokenizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import html

from collections import Counter
import spacy

nlp = spacy.load("en_core_web_sm")

import requests

# Initialize YouTube API
def initialize_youtube():
    api_key = os.getenv('YOUTUBE_API_KEY')
    if not api_key:
        raise ValueError("YouTube API key not found. Please set the YOUTUBE_API_KEY environment variable.")
    return build('youtube', 'v3', developerKey=api_key)

# Fetch YouTube data
def fetch_youtube_data(youtube, keyword, rate_limit):
    search_response = youtube.search().list(
        part="snippet",
        q=keyword,
        type="video",
        maxResults=rate_limit
    ).execute()

    videos = []
    for item in search_response.get("items", []):
        video_title = item["snippet"]["title"]
        video_id = item["id"]["videoId"]
        channel_title = item["snippet"]["channelTitle"]
        published_at = item["snippet"]["publishedAt"]

        video_response = youtube.videos().list(
            part="statistics",
            id=video_id
        ).execute()

        view_count = int(video_response["items"][0]["statistics"].get("viewCount", 0))
        like_count = int(video_response["items"][0]["statistics"].get("likeCount", 0))
        comment_count = int(video_response["items"][0]["statistics"].get("commentCount", 0))

        videos.append({
            'video_title': video_title,
            'video_id': video_id,
            'channel_title': channel_title,
            'published_at': published_at,
            'view_count': view_count,
            'like_count': like_count,
            'comment_count': comment_count
        })

    return videos




# Analyze sentiment of comments
def analyze_sentiment(comments):
    analyzer = SentimentIntensityAnalyzer()
    sentiment_scores = {'positive': 0, 'neutral': 0, 'negative': 0}
    
    for comment in comments:
        sentiment = analyzer.polarity_scores(comment)
        if sentiment['compound'] >= 0.05:
            sentiment_scores['positive'] += 1
        elif sentiment['compound'] <= -0.05:
            sentiment_scores['negative'] += 1
        else:
            sentiment_scores['neutral'] += 1
    
    return sentiment_scores

# Analyze common themes in comments
def analyze_common_themes(comments):
    # Process each comment and extract nouns and noun phrases
    themes = []
    for comment in comments:
        doc = nlp(comment)
        for token in doc:
            if token.pos_ in ["NOUN", "PROPN"]:
                themes.append(token.text)
            elif token.pos_ == "NOUN_CHUNK":
                themes.append(token.text)
    
    # Count occurrences of themes
    theme_counter = Counter(themes)
    
    # Get the most common themes
    common_themes = theme_counter.most_common(3)  # Get top 3 most common themes
    return [theme[0] for theme in common_themes]

    return common_themes

# Compare with reliable sources
def compare_with_reliable_sources(videos, keyword):
    # Define reliable sources or APIs to verify information
    reliable_sources = {
        "Wikipedia": "https://en.wikipedia.org/wiki/Main_Page",
        "BBC News": "https://www.bbc.com/news",
        "The New York Times": "https://www.nytimes.com/",
        "The Guardian": "https://www.theguardian.com/international",
        "CNN": "https://edition.cnn.com/",
        "Reuters": "https://www.reuters.com/",
        "National Geographic": "https://www.nationalgeographic.com/",
        "NPR (National Public Radio)": "https://www.npr.org/",
        "Associated Press": "https://apnews.com/",
        "TIME": "https://time.com/",
        "Scientific American": "https://www.scientificamerican.com/",
        "Nature": "https://www.nature.com/",
        "Harvard Health Publishing": "https://www.health.harvard.edu/",
        "Smithsonian Magazine": "https://www.smithsonianmag.com/",
        "The Economist": "https://www.economist.com/"
    }
    
    comparison_result = ""
    
    # Iterate through videos and check against reliable sources
    for video in videos:
        # Extract relevant information from the video
        video_title = video['video_title']
        # Example: Extract video description or transcript
        video_description = video.get('video_description', '')  # Get video description, handle if it doesn't exist
        
        # Iterate through reliable sources
        for source_name, source_url in reliable_sources.items():
            # Example: Query the reliable source API with relevant information
            response = requests.get(source_url, params={'query': video_description})
            if response.status_code == 200:
                try:
                    response_json = response.json()  # Try to parse JSON response
                    if response_json.get('reliability') == 'high':
                        comparison_result += f"Video '{video_title}' aligns with {source_name}.\n"
                    else:
                        comparison_result += f"Warning: Video '{video_title}' may contain unreliable information according to {source_name}.\n"
                except json.JSONDecodeError:
                    comparison_result += f"Error: Unable to parse response from {source_name} as JSON.\n"
            else:
                comparison_result += f"Error: Unable to retrieve data from {source_name}.\n"
    
    return comparison_result

# Analyze trend over time
def analyze_trend_over_time(videos):
    # Placeholder implementation for trend analysis
    views = [video['view_count'] for video in videos]
    likes = [video['like_count'] for video in videos]
    comments = [video['comment_count'] for video in videos]

    # Calculate average views, likes, and comments
    avg_views = sum(views) / len(views)
    avg_likes = sum(likes) / len(likes)
    avg_comments = sum(comments) / len(comments)

    # Check if there's an increasing trend in views, likes, and comments over time
    increasing_views = all(views[i] >= views[i - 1] for i in range(1, len(views)))
    increasing_likes = all(likes[i] >= likes[i - 1] for i in range(1, len(likes)))
    increasing_comments = all(comments[i] >= comments[i - 1] for i in range(1, len(comments)))

    # Construct trend analysis message
    trend_analysis = "Trend Analysis:\n"
    trend_analysis += f"Average Views: {avg_views}\n"
    trend_analysis += f"Average Likes: {avg_likes}\n"
    trend_analysis += f"Average Comments: {avg_comments}\n"
    trend_analysis += "There is "
    if increasing_views:
        trend_analysis += "an increasing trend in views, "
    if increasing_likes:
        trend_analysis += "an increasing trend in likes, "
    if increasing_comments:
        trend_analysis += "an increasing trend in comments, "
    if not (increasing_views or increasing_likes or increasing_comments):
        trend_analysis += "no significant trend observed in views, likes, or comments over time."

    return trend_analysis


# Analyze fake news
def analyze_fake_news(youtube, keyword, rate_limit):
    videos = fetch_youtube_data(youtube, keyword, rate_limit)

    comments = []
    for video in videos:
        video_id = video['video_id']
        comment_response = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            searchTerms=keyword,
            maxResults=rate_limit
        ).execute()

        if "items" in comment_response:
            for comment in comment_response["items"]:
                comment_text = comment["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
                comment_text = html.unescape(comment_text)
                comments.append(comment_text)

    sentiment_scores = analyze_sentiment(comments)
    common_themes = analyze_common_themes(comments)
    comparison_result = compare_with_reliable_sources(videos, keyword)
    trend_analysis = analyze_trend_over_time(videos)

    return videos, comments, sentiment_scores, common_themes, comparison_result, trend_analysis

# Draw conclusions based on analysis results
def draw_conclusions(videos, comments, sentiment_scores, common_themes, comparison_result, trend_analysis):
    total_comments = sum(sentiment_scores.values())

    if total_comments == 0:
        return "No comments found.", 0, 0, [], "", ""

    # 1. Sentiment Analysis
    positive_percentage = (sentiment_scores['positive'] / total_comments) * 100
    negative_percentage = (sentiment_scores['negative'] / total_comments) * 100

    if positive_percentage > negative_percentage:
        conclusion_sentiment = "positive"
    elif negative_percentage > positive_percentage:
        conclusion_sentiment = "negative"
    else:
        conclusion_sentiment = "neutral"

    # 2. Engagement Metrics
    total_views = sum(int(video['view_count']) for video in videos)
    total_likes = sum(int(video['like_count']) for video in videos)
    average_views = total_views / len(videos)
    average_likes = total_likes / len(videos)

    return conclusion_sentiment, average_views, average_likes, common_themes, comparison_result, trend_analysis

# Example usage
def main():
    # Get input from the user
    keyword = input("Enter the keyword to search for: ")
    rate_limit = int(input("Enter the maximum number of videos to fetch: "))

    youtube = initialize_youtube()
    videos, comments, sentiment_scores, common_themes, comparison_result, trend_analysis = analyze_fake_news(youtube, keyword, rate_limit)

    # Output videos and usernames
    for video in videos:
        video_title = video['video_title']
        video_id = video['video_id']
        video_link = f"https://www.youtube.com/watch?v={video_id}"
        channel_title = video['channel_title']
        print("Video Title:", video_title)
        print("Channel Title:", channel_title)
        print("Video Link:", video_link)
        print()

    # Draw conclusions
    conclusion_sentiment, average_views, average_likes, common_themes, comparison_result, trend_analysis = draw_conclusions(videos, comments, sentiment_scores, common_themes, comparison_result, trend_analysis)
    print("Conclusion Sentiment:", conclusion_sentiment)
    print("Average Views:", average_views)
    print("Average Likes:", average_likes)
    print("Common Themes:", common_themes)
    print("Comparison Result:", comparison_result)
    print("Trend Analysis:", trend_analysis)

if __name__ == "__main__":
    main()
