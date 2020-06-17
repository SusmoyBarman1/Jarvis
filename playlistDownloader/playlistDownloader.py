# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 22:19:33 2020

@author: SuSu
"""

# import the module
from pytube import Playlist
 
# save location
SAVE_PATH = "playlist/"
 
# playlist link
link = "https://www.youtube.com/watch?v=Z7YM-HAz-IY&list=PLhA3b2k8R3t2Ng1WW_7MiXeh1pfQJQi_P"
 
# download the playlist
try:
    print("Starting playlist download...")
    playlist = Playlist(link)
 
    # for each video in the playlist
    counter = 1
    for video in playlist.videos:
        print("Downloading video ", str(counter))
        video.streams.get_highest_resolution().download(SAVE_PATH)
        counter += 1
 
    # download complete
    print("Download completed!")
 
except:
    print("Connection error...")