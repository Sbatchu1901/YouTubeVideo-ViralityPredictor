# Note: You may need to install the 'requests' library to use the Perspective API.
# pip install requests
import os
import json
import time
import isodate
import requests
import pandas as pd
from textblob import TextBlob
from datetime import datetime, timedelta, timezone
from tqdm import tqdm
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError


# --------------------
#  Set your API keys
# --------------------
try:
    # YouTube API key - used for data collection
    YOUTUBE_API_KEY = os.environ.get('YOUTUBE_API_KEY', 'AIzaSyDQTHZd_yS55pqLW8wYYKQwkzux4J_4i3o')
    if YOUTUBE_API_KEY == 'AIzaSyDQTHZd_yS55pqLW8wYYKQwkzux4J_4i3o':
        print("⚠️ Warning: Please set your YouTube API key.")
    youtube = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)

    # Perspective API key - used for comment toxicity analysis
    PERSPECTIVE_API_KEY = os.environ.get('PERSPECTIVE_API_KEY', 'AIzaSyAif4SXm5J8b9SUCCU1os2XqRg5ZnGXcPY')
    if PERSPECTIVE_API_KEY == 'AIzaSyAif4SXm5J8b9SUCCU1os2XqRg5ZnGXcPY':
        print("⚠️ Warning: Please set your Perspective API key.")
except Exception as e:
    print(f"❌ Error initializing API clients: {e}")
    youtube = None
    PERSPECTIVE_API_KEY = None
    
# ---------------------------------------------
#  Fetch video IDs based on query + pagination + date range
# ---------------------------------------------
def search_videos(query, max_results=1000, days_back=7):
    """
    Search for videos using a query and return a list of video IDs.
    """
    if not youtube:
        return []

    video_ids = []
    next_page_token = None
    fetched = 0
    published_after = (datetime.now(timezone.utc) - timedelta(days=days_back)).isoformat()

    try:
        while fetched < max_results:
            results_to_fetch = min(50, max_results - fetched)
            if results_to_fetch <= 0:
                break
            response = youtube.search().list(
                q=query, part='id,snippet', maxResults=results_to_fetch, type='video',
                order='date', publishedAfter=published_after, pageToken=next_page_token
            ).execute()
            for item in response.get('items', []):
                if 'videoId' in item['id']:
                    video_ids.append(item['id']['videoId'])
                    fetched += 1
            next_page_token = response.get('nextPageToken')
            if not next_page_token or fetched >= max_results:
                break
            time.sleep(1)
        print(f"🔎 Found {len(video_ids)} videos for query '{query}' published in last {days_back} days.")
        return video_ids
    except HttpError as e:
        print(f"❌ An API error occurred during video search: {e}")
        return video_ids

# -------------------------------
#  Get metadata for a list of videos (batched)
# -------------------------------
def get_video_metadata_batch(video_ids):
    """
    Fetches metadata for a list of video IDs in a single batched API call (max 50 IDs).
    """
    if not youtube or not video_ids:
        return []
    metadata_list = []
    video_id_string = ','.join(video_ids)
    try:
        response = youtube.videos().list(
            part='snippet,statistics,contentDetails', id=video_id_string
        ).execute()
        for item in response.get('items', []):
            duration_iso = item['contentDetails'].get('duration', 'PT0S')
            duration_seconds = isodate.parse_duration(duration_iso).total_seconds()
            publish_dt = datetime.strptime(item['snippet']['publishedAt'], '%Y-%m-%dT%H:%M:%SZ').replace(tzinfo=timezone.utc)
            days_since_publish = (datetime.now(timezone.utc) - publish_dt).days or 1
            views = int(item['statistics'].get('viewCount', 0))
            metadata = {
                'video_id': item['id'],
                'title': item['snippet']['title'],
                'description': item['snippet']['description'],
                'publish_date': item['snippet']['publishedAt'],
                'channel_title': item['snippet']['channelTitle'],
                'category_id': item['snippet']['categoryId'],
                'duration_seconds': duration_seconds,
                'views': views,
                'likes': int(item['statistics'].get('likeCount', 0)),
                'comments_count': int(item['statistics'].get('commentCount', 0)),
                'days_since_publish': days_since_publish,
                'views_per_day': views / days_since_publish if days_since_publish > 0 else 0
            }
            metadata_list.append(metadata)
    except HttpError as e:
        print(f"❌ An API error occurred during metadata retrieval: {e}")
    return metadata_list

# ------------------------------------
#  Get top-level comments and replies
# ------------------------------------
def get_comments_with_replies(video_id, max_comments=30):
    """
    Fetches top-level comments for a video and returns them as a list of dictionaries.
    """
    if not youtube:
        return []
    all_comments = []
    try:
        response = youtube.commentThreads().list(
            part='snippet', videoId=video_id, maxResults=min(max_comments, 100),
            textFormat='plainText'
        ).execute()
        for item in response.get('items', []):
            comment = {
                'text': item['snippet']['topLevelComment']['snippet']['textDisplay'],
                'reply_count': item['snippet']['totalReplyCount'],
                'toxicity_score': None  # Placeholder to be filled by Perspective API
            }
            all_comments.append(comment)
    except HttpError as e:
        print(f"❌ Could not fetch comments for video {video_id}: {e}")
    return all_comments

# ------------------------------------------------
#  Analyze comments with the Perspective API
# ------------------------------------------------
def get_toxicity_scores(comments):
    """
    Sends a list of comments to the Perspective API and returns their toxicity scores.
    Includes a robust retry mechanism for rate limits.
    """
    if not PERSPECTIVE_API_KEY or not comments:
        return [None] * len(comments)

    url = f'https://commentanalyzer.googleapis.com/v1alpha1/comments:analyze?key={PERSPECTIVE_API_KEY}'
    scores = []
    
    # Process comments one by one to better manage rate limits
    for comment_text in comments:
        payload = {
            'comment': {'text': comment_text},
            'languages': ['en'],
            'requestedAttributes': {'TOXICITY': {}}
        }
        retries = 0
        while retries < 5:
            try:
                response = requests.post(url, json=payload)
                response.raise_for_status() # Raise an exception for bad status codes
                result = response.json()
                score = result['attributeScores']['TOXICITY']['summaryScore']['value']
                scores.append(score)
                break # Success, move to the next comment
            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 429: # Too Many Requests
                    wait_time = 2**retries # Exponential backoff
                    print(f"❌ Rate limit hit, retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                    retries += 1
                else:
                    print(f"❌ An unexpected API error occurred: {e}")
                    scores.append(None)
                    break
            except requests.exceptions.RequestException as e:
                print(f"❌ A request error occurred with the Perspective API: {e}")
                scores.append(None)
                break
        else:
            # This block runs if the `while` loop completes without a `break` (i.e., after 5 retries)
            print(f"❌ Failed to get toxicity score for a comment after {retries} retries.")
            scores.append(None)

    return scores

# ----------------------
#  Full data collection pipeline
# ----------------------
def fetch_and_save_data(query='machine learning tutorial', num_videos=1000, days_back=30):
    """
    The main pipeline to search for videos, fetch their metadata, comments,
    and toxicity scores, and then save the data to a JSON file.
    """
    if not youtube or not PERSPECTIVE_API_KEY:
        print(" API clients are not initialized. Exiting.")
        return

    video_ids = search_videos(query, max_results=num_videos, days_back=days_back)
    all_data = []

    # Batch video IDs into chunks of 50 for more efficient metadata fetching
    for i in tqdm(range(0, len(video_ids), 50), desc="Fetching YouTube Data in batches"):
        batch_ids = video_ids[i:i+50]
        batch_metadata = get_video_metadata_batch(batch_ids)

        for meta in batch_metadata:
            # We fetch comments for each video individually
            comments = get_comments_with_replies(meta['video_id'])
            
            # Extract comment text to send to Perspective API
            comment_texts = [c['text'] for c in comments]
            toxicity_scores = get_toxicity_scores(comment_texts)

            # Add the toxicity score back to each comment object
            for j, score in enumerate(toxicity_scores):
                if j < len(comments):
                    comments[j]['toxicity_score'] = score
            
            meta['comments_data'] = comments
            all_data.append(meta)

        time.sleep(1)

    os.makedirs('1_data_collection/raw_data', exist_ok=True)
    json_path = '1_data_collection/raw_data/youtube_data.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(all_data, f, indent=2, ensure_ascii=False)

    print(f"\n Saved {len(all_data)} videos to {json_path}")
    return json_path


# ----------------------
#  Preprocessing and Feature Engineering
# ----------------------
def load_raw_data(json_path='1_data_collection/raw_data/youtube_data.json'):
    """Loads the raw JSON data from the specified path."""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def sentiment_score(text):
    """Calculates the sentiment polarity of a given text."""
    return TextBlob(str(text)).sentiment.polarity

def feature_engineering(raw_data):
    """
    Performs feature engineering on the raw YouTube data.
    
    Args:
        raw_data (list): A list of dictionaries containing raw video data.

    Returns:
        pd.DataFrame: A pandas DataFrame with engineered features.
    """
    rows = []
    for video in raw_data:
        # Basic metadata
        vid = video['video_id']
        title = video.get('title', '')
        description = video.get('description', '')
        views = video.get('views', 0)
        likes = video.get('likes', 0)
        comments_count = video.get('comments_count', 0)
        duration = video.get('duration_seconds', 0)
        publish_date = video.get('publish_date', None)
        
        if publish_date:
            publish_dt = datetime.strptime(publish_date, '%Y-%m-%dT%H:%M:%SZ').replace(tzinfo=timezone.utc)
            days_since_publish = (datetime.now(timezone.utc) - publish_dt).days or 1
        else:
            days_since_publish = 1
        
        views_per_day = views / days_since_publish if days_since_publish > 0 else 0
        
        # Text-based features
        title_len = len(title)
        desc_len = len(description)
        title_sentiment = sentiment_score(title)
        desc_sentiment = sentiment_score(description)
        
        # New: Calculate toxicity features from comments_data
        comments_data = video.get('comments_data', [])
        toxicity_scores = [c.get('toxicity_score') for c in comments_data if c.get('toxicity_score') is not None]
        mean_toxicity = sum(toxicity_scores) / len(toxicity_scores) if toxicity_scores else 0
        pct_toxic_comments = len([s for s in toxicity_scores if s > 0.8]) / len(toxicity_scores) if toxicity_scores else 0
        
        # Viral label example: views_per_day > 1000 (adjust as needed)
        viral_label = 1 if views_per_day > 1000 else 0
        
        rows.append({
            'video_id': vid,
            'views': views,
            'likes': likes,
            'comments_count': comments_count,
            'duration_seconds': duration,
            'days_since_publish': days_since_publish,
            'views_per_day': views_per_day,
            'title_len': title_len,
            'desc_len': desc_len,
            'title_sentiment': title_sentiment,
            'desc_sentiment': desc_sentiment,
            'mean_toxicity': mean_toxicity,
            'pct_toxic_comments': pct_toxic_comments,
            'viral_label': viral_label
        })
    return pd.DataFrame(rows)

# ----------------------
#  Run the full pipeline
# ----------------------
if __name__ == '__main__':
    # Step 1: Collect raw data
    # Here are some trending topics you can use for your search query.
    # Just uncomment the one you want to use and comment out the current one.
    json_output_path = fetch_and_save_data(
         query="machine learning tutorial",
        num_videos=1000,
        days_back=30
    )

    # Step 2: Perform feature engineering
    if os.path.exists(json_output_path):
        os.makedirs('2_preprocessing', exist_ok=True)
        raw_data = load_raw_data(json_output_path)
        df_features = feature_engineering(raw_data)
        df_features.to_csv('2_preprocessing/clean_data.csv', index=False)
        print(" Feature engineering complete, saved to 2_preprocessing/clean_data.csv")
    else:
        print(" Raw data JSON file was not found. Cannot perform feature engineering.")
    
    
    
    
    
    
    
    
    
    