#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install lyricsgenius')


# In[ ]:





# In[3]:


import lyricsgenius

# Instantiate a Genius object with your access token
genius = lyricsgenius.Genius("SRpQ9wybL3JRoWDp6G85dRwYurIzGp_zQppA_Jn1Z4RR6PBMYu04QK5lgrO0yvNh", timeout=15)

# Search for Taylor Swift
artist = genius.search_artist("Taylor Swift", max_songs=50, sort="popularity")

# Open a file to write the song titles and lyrics
with open('taylor_swift_lyrics.txt', 'w') as file:
    # For each song, write the song title and lyrics to the file
    for song in artist.songs:
        file.write(f'Song Title: {song.title}\n')
        file.write(f'Lyrics:\n{song.lyrics}\n')
        file.write('-'*50 + '\n')


# In[4]:


import lyricsgenius
import time

genius = lyricsgenius.Genius("SRpQ9wybL3JRoWDp6G85dRwYurIzGp_zQppA_Jn1Z4RR6PBMYu04QK5lgrO0yvNh", timeout=15)

def fetch_songs(artist_name, max_songs=None, sort="popularity"):
    attempts = 0
    while attempts < 5:
        try:
            artist = genius.search_artist(artist_name, max_songs=max_songs, sort=sort)
            return artist
        except lyricsgenius.Timeout as e:
            print(f"Timeout error, retrying... ({attempts+1})")
            time.sleep(10)  # Sleep for 10 seconds before retrying
            attempts += 1
    print("Failed to fetch songs after multiple attempts")
    return None

artist = fetch_songs("Taylor Swift")

if artist:
    with open('taylor_swift_lyrics.txt', 'w') as file:
        for song in artist.songs:
            file.write(f'Song Title: {song.title}\n')
            file.write(f'Lyrics:\n{song.lyrics}\n')
            file.write('-'*50 + '\n')


# In[7]:


import lyricsgenius
import time

genius = lyricsgenius.Genius("SRpQ9wybL3JRoWDp6G85dRwYurIzGp_zQppA_Jn1Z4RR6PBMYu04QK5lgrO0yvNh", timeout=15)

def fetch_songs(artist_name, max_songs=200, sort="popularity"):
    attempts = 0
    while attempts < 5:
        try:
            artist = genius.search_artist(artist_name, max_songs=max_songs, sort=sort)
            return artist
        except lyricsgenius.Timeout as e:
            print("Timeout error, retrying... ({attempts+1})")
            time.sleep(10)  # Sleep for 10 seconds before retrying
            attempts += 1
    print("Failed to fetch songs after multiple attempts")
    return None

artist = fetch_songs("Taylor Swift")

if artist:
    with open('taylor_swift_lyrics.txt', 'w') as file:
        for song in artist.songs:
            file.write(f'Song Title: {song.title}\n')
            file.write(f'Lyrics:\n{song.lyrics}\n')
            file.write('-'*50 + '\n')
            file.flush()  # Force write to disk after each song


# In[8]:


import re

def clean_text(text):
    # Remove lines with 'Embed', 'Contributors', 'Translations' etc.
    text = re.sub(r'.*Contributors.*|.*Translations.*|.*Embed.*|.*See Taylor Swift Live.*|.*You might also like.*', '', text)
    
    # Remove lines that are just 'Lyrics:'
    text = re.sub(r'^Lyrics:$', '', text)
    
    # Remove lines that are just '--------------------------------------------------'
    text = re.sub(r'^-{2,}$', '', text)
    
    return text

# Open the input file and read the text
with open('taylor_swift_lyrics.txt', 'r') as f:
    text = f.read()

# Apply the function to the text
cleaned_text = "\n".join(clean_text(line) for line in text.split("\n"))

# Write the cleaned text to a new file
with open('taylor_swift_lyrics_clean.txt', 'w') as f:
    f.write(cleaned_text)


# In[ ]:




