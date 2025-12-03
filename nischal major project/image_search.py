import requests
from bs4 import BeautifulSoup
import json

def get_google_images(query, num_images=3):
    """Fetch image URLs from Google Images"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        url = f"https://www.google.com/search?q={query}&tbm=isch"
        response = requests.get(url, headers=headers, timeout=5)
        
        # Extract image URLs from Google Images
        images = []
        soup = BeautifulSoup(response.content, 'html.parser')
        img_tags = soup.find_all('img', limit=num_images+1)
        
        for img in img_tags[1:num_images+1]:
            if 'src' in img.attrs:
                images.append(img['src'])
        
        return images[:num_images]
    except:
        return []

def get_unsplash_images(query, num_images=3):
    """Fetch images from Unsplash API"""
    try:
        url = f"https://api.unsplash.com/search/photos"
        params = {
            'query': query,
            'per_page': num_images,
            'client_id': 'YOUR_UNSPLASH_ACCESS_KEY'
        }
        response = requests.get(url, params=params, timeout=5)
        data = response.json()
        
        images = []
        if 'results' in data:
            for result in data['results']:
                images.append(result['urls']['regular'])
        
        return images
    except:
        return []

def get_pexels_images(query, num_images=3):
    """Fetch images from Pexels API"""
    try:
        url = "https://api.pexels.com/v1/search"
        headers = {
            'Authorization': 'YOUR_PEXELS_API_KEY'
        }
        params = {
            'query': query,
            'per_page': num_images
        }
        response = requests.get(url, headers=headers, params=params, timeout=5)
        data = response.json()
        
        images = []
        if 'photos' in data:
            for photo in data['photos']:
                images.append(photo['src']['medium'])
        
        return images
    except:
        return []

def get_disaster_images(disaster_type):
    """Get images for a specific disaster type"""
    search_queries = {
        'biological and chemical pandemic': 'pandemic safety health',
        'cyclone': 'cyclone storm damage',
        'drought': 'drought dry land water scarcity',
        'earthquake': 'earthquake damage building collapse',
        'flood': 'flood water disaster',
        'landslide': 'landslide mountain slope',
        'tsunami': 'tsunami wave ocean',
        'wildfire': 'wildfire forest fire smoke'
    }
    
    query = search_queries.get(disaster_type, disaster_type)
    
    images = {
        'google': get_google_images(query, 2),
        'unsplash': get_unsplash_images(query, 2),
        'pexels': get_pexels_images(query, 2)
    }
    
    return images
