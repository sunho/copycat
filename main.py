from copycat import copycat
import sys
import threading

with open('api.txt', 'r') as file:
    API_KEY = file.read().replace('\n', '')

with open('secret.txt', 'r') as file:
    API_SECRET = file.read().replace('\n', '')



status_monitor = copycat.StatusMonitor()
executor = copycat.BinanceExecutor(API_KEY, API_SECRET, status_monitor)
decider = copycat.CopycatDecider(0.0006, 0.002, 0.008, 20, status_monitor)
engine = copycat.Engine(executor, decider, status_monitor)

def run_feed():
    while True:
        url = copycat.get_latest_live_stream("https://www.youtube.com/@wedomnbro")
        print("livestream url: ", url)
        feed = copycat.VideoFeed(engine.on_position, 600, 3600)
        feed.start(url)

thread = threading.Thread(target=status_monitor.start)
thread.daemon = True
thread.start()

run_feed()

