from copycat import copycat

with open('api.txt', 'r') as file:
    API_KEY = file.read().replace('\n', '')

with open('secret.txt', 'r') as file:
    API_SECRET = file.read().replace('\n', '')

youtube_live_url = 'https://www.youtube.com/watch?v=Fxs4CVvNHwE'
engine = copycat.Engine(API_KEY, API_SECRET)
engine.start(youtube_live_url)
