from abc import abstractmethod
from collections import deque
from io import BytesIO
from rapidocr_onnxruntime import RapidOCR
from dataclasses import dataclass
import re

import time

import cv2
import subprocess
import cv2
from PIL import Image

from binance.client import Client
from binance.enums import SIDE_BUY, SIDE_SELL
import threading

from rich.console import Console
from rich.table import Table
from rich.live import Live
from rich.layout import Layout
from rich.panel import Panel
from rich.text import Text

import requests
from bs4 import BeautifulSoup

def get_latest_live_stream(channel_url):
    return f"{channel_url}/live"
    try:
        # Fetch the channel's videos page
        response = requests.get(f"{channel_url}/live")
        response.raise_for_status()

        # Parse the page
        soup = BeautifulSoup(response.text, 'html.parser')

        # Look for the live stream link
        live_stream_tag = soup.find('link', {'rel': 'canonical'})
        if live_stream_tag:
            live_stream_url = live_stream_tag.get('href', '')
            if '/live' in live_stream_url and "watch" in live_stream_url:
                return live_stream_url
            else:
                return "No live stream is currently active."
        else:
            return "Unable to find a live stream link on the page."

    except requests.RequestException as e:
        return f"An error occurred: {e}"

def get_live_stream_hls_url(video_url):
    command = [
        "yt-dlp",
        "-g", 
        video_url
    ]
    try:
        hls_url = subprocess.check_output(command, text=True).strip()
        return hls_url
    except subprocess.CalledProcessError as e:
        print("Error fetching the HLS URL:", e)
        return None

RATIO = 0.0002

@dataclass
class Position:
    symbol: str
    is_short: bool
    position: int
    avg_price: float
    leverage: int 


class VideoFeed:
    def __init__(self, on_position, interval, duration):
        self.ocr_engine = RapidOCR()
        self.interval = interval
        self.duration = duration
        self.on_position = on_position

    def start(self, url):
        hls_url = get_live_stream_hls_url(url)
        cap = cv2.VideoCapture(hls_url)

        if not cap.isOpened():
            print("Error: Cannot open the video stream.")
            return
        
        cnt = 0
        for _ in range(self.duration):
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break

            frame = self.crop_frame(frame)
            if cnt % self.interval == 0:
                thread = threading.Thread(target=self.feed_frame, args=(frame,))
                thread.daemon = True
                thread.start()
            cnt+=1

            # cv2.imshow('Live Frame', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        #cv2.destroyAllWindows()

    def crop_left_bottom(self, frame, crop_width, crop_height, crop_height2):
        height, width, _ = frame.shape
        x_start = 0
        y_start = height - crop_height
        return frame[y_start:height-crop_height2, x_start:crop_width]

    def frame_to_png_buffer(self, cropped_frame):
        image = Image.fromarray(cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2RGB))
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        buffer.seek(0)
        return buffer.read()

    def crop_frame(self, frame):
        return self.crop_left_bottom(frame, 750, 200, 80)

    def extract_state(self, img: bytes) -> Position:
        position = Position("BTCUSDT", True, 0, 0, 0)
        result, _ = self.ocr_engine(img)
        for (_, text, _) in result:
            if "Short" in text or "short" in text:
                position.is_short = True
            elif "Long" in text or "long" in text:
                position.is_short = False
            pattern = r"(\d+)[Xx]"
            matches = re.findall(pattern, text)
            if matches:
                position.leverage = int(matches[0])
            pattern = r"(\d+).*BTC"
            matches = re.findall(pattern, text)
            if matches:
                position.position = float(matches[0])

            try:
                x = float(text.replace(",", ""))
                position.avg_price = x
            except ValueError:
                pass
        return position
    
    def feed_frame(self, frame):
        buf = self.frame_to_png_buffer(frame)
        self.on_position(self.extract_state(buf))


class StatusMonitor:
    def __init__(self):
        layout = Layout()
        layout.split_column(
            Layout(name="upper", size=6),
            Layout(name="lower"),
            Layout(name="stdout")
        )

        self.my_position = None
        self.his_position = None
        self.price = 0
        self.update_counter = 0
        self.layout = layout
        self.output_buffer = deque(maxlen=16)
    
    def append_output(self, text):
        self.output_buffer.append(text)

    def create_output_console(self):
        # Combine the output buffer into a single Text object
        output_text = Text()
        for line in self.output_buffer:
            output_text.append(line+'\n')
        
        output_panel = Panel(
            output_text,
            title="Output Console",
            border_style="bright_green",
            padding=(1, 2)
        )
        return output_panel
            
    def start(self):
        with Live(self.layout, refresh_per_second=1):
            while True:
                self.render_header()
                self.rerender_table()
                self.render_stdout()
                time.sleep(1)

    def render_stdout(self):
        self.layout["stdout"].update(self.create_output_console())

    def increment_counter(self):   
        self.update_counter += 1

    def update_price(self, price):
        self.price = price

    def report_my_position(self, position):
        self.my_position = position
    
    def report_his_position(self, position):
        self.his_position = position
    
    def render_header(self):
        header_text =  Text("Live Trading Engine Status\n")
        header_text.append("Current Price of BTCUSDT: ", style="bold green")
        header_text.append(f"{self.price}\n", style="bold yellow")
        header_text.append("Update Counter: ", style="bold cyan")
        header_text.append(f"{self.update_counter}", style="bold magenta")
        self.layout["upper"].update(Panel(header_text, title="Header"))
    
    def rerender_table(self):
        table = Table(title="Trading Status", expand=True)
        table.add_column("Symbol", justify="left", style="cyan", no_wrap=True)
        table.add_column("Side", justify="left", style="magenta")
        table.add_column("Position", justify="right", style="green")
        table.add_column("Avg Price", justify="right", style="yellow")
        table.add_column("Leverage", justify="right", style="blue")
        table.add_column("PNL", justify="right", style="green")
        table.add_column("PNL_pct", justify="right", style="green")
        table.add_column("Owner", justify="right", style="blue")
        def get_pnl(position):
            return (self.price - position.avg_price) * position.position * (-1 if position.is_short else 1)
        def get_pnlpct(position):
            return (self.price - position.avg_price) / position.avg_price * 100 * (-1 if position.is_short else 1) * position.leverage
        def format_float(x):
            return "{:.4f}".format(x)
        if self.my_position:
            table.add_row(
                self.my_position.symbol,
                "Short" if self.my_position.is_short else "Long",
                str(self.my_position.position),
                str(self.my_position.avg_price),
                str(self.my_position.leverage),
                format_float(get_pnl(self.my_position)),
                format_float(get_pnlpct(self.my_position))+"%",
                "You"
            )
        if self.his_position:
            table.add_row(
                self.his_position.symbol,
                "Short" if self.his_position.is_short else "Long",
                str(self.his_position.position),
                str(self.his_position.avg_price),
                str(self.his_position.leverage),
                format_float(get_pnl(self.his_position)),
                format_float(get_pnlpct(self.his_position))+"%",
                "Him"
            )
        self.layout["lower"].update(Panel(table, title="Positions"))


class BinanceOrderMaker:
    def __init__(self, api_key, api_secret, status_monitor):
        self.client = Client(api_key, api_secret)
        self.status_monitor = status_monitor

    def get_positions(self) -> list[Position]:
        positions = []
        try:
            account_info = self.client.futures_account()
            for position_data in account_info["positions"]:
                if float(position_data["positionAmt"]) != 0:
                    position = Position(
                        symbol=position_data["symbol"],
                        is_short=float(position_data["positionAmt"]) < 0,
                        position=abs(float(position_data["positionAmt"])),
                        avg_price=float(position_data["entryPrice"]),
                        leverage=int(position_data["leverage"])
                    )
                    positions.append(position)
        except Exception as e:
            self.status_monitor.append_output(f"An error occurred: {e}")
        
        return positions

    def create_isolated_order(self, position: Position):
        symbol = "BTCUSDT"
        side = position.is_short and SIDE_SELL or SIDE_BUY
        qty = position.position
        if qty == 0:
            return
        self.status_monitor.append_output(f"Creating order: {position}")

        try:
            self.client.futures_change_margin_type(symbol=symbol, marginType="ISOLATED")
        except Exception:
            pass

        try:
            self.client.futures_change_leverage(symbol=symbol, leverage=position.leverage)
        except Exception:
            pass

        try:
            order = self.client.futures_create_order(
                symbol=symbol,
                side=side,
                type="MARKET",
                quantity=qty,
            )
            self.status_monitor.append_output(f"Order created: {order}")
        except Exception as e:
            self.status_monitor.append_output(f"An error occurred: {e}")
    
    def close_all(self):
        positions = self.get_isolated_positions()
        for position in positions:
            position.is_short = not position.is_short
            self.create_isolated_order(position)
        
    def get_price(self, symbol):
        price = float(self.client.get_symbol_ticker(symbol=symbol)["price"])
        return price
    

class ExecutorBase:
    @abstractmethod
    def exec_position(self, position):
        pass

class BinanceExecutor(ExecutorBase):
    def __init__(self, api_key, api_secret, status_monitor):
        self.binance = BinanceOrderMaker(api_key, api_secret, status_monitor)

    def get_positions(self) -> list[Position]:
        return self.binance.get_positions()

    def exec_position(self, position):
        positions = self.get_positions()
        if len(positions) == 0:
            self.binance.create_isolated_order(position)
        else:
            cur = positions[0]
            qty_diff = (-1 if position.is_short else 1) * position.position - (-1 if cur.is_short else 1) * cur.position
            if qty_diff > 0:
                new_order = Position(cur.symbol, False, qty_diff, 0, position.leverage)
                self.binance.create_isolated_order(new_order)
            else:
                new_order = Position(cur.symbol, True, abs(qty_diff), 0, position.leverage)
                self.binance.create_isolated_order(new_order)

    def get_price(self, symbol):
        return self.binance.get_price(symbol)

class DummyExecutor(ExecutorBase):
    def get_positions(self) -> list[Position]:
        return []

    def exec_position(self, position):
        print("Execute position: ", position)

    def get_price(self, symbol):
        return 0

class Decider:
    @abstractmethod
    def decide_position(self, my_pos, his_pos):
        pass

TICK = 0.001
class CopycatDecider(Decider):
    def __init__(self, ratio, min_qty, max_qty, leverage, status_monitor):
        self.cnt = 0
        self.ratio = ratio
        self.min_qty = min_qty
        self.max_qty = max_qty
        self.leverage = leverage
        self.status_monitor = status_monitor

    def decide_position(self, my_pos, his_pos):
        if his_pos.position == 0:
            if self.cnt < 20:
                self.cnt += 1
                return my_pos
            self.cnt = 0
        new_pos = Position("BTCUSDT", his_pos.is_short, 0, 0, self.leverage)
        if his_pos:
            new_pos.position = (his_pos.position * self.ratio)//TICK*TICK
            new_pos.position = min(new_pos.position, self.max_qty)
            new_pos.position = max(new_pos.position, self.min_qty)
            self.status_monitor.append_output(f"New position to set: {new_pos}")
            return new_pos
        return my_pos

class Engine:
    def __init__(self, executor, decider, status_monitor):
        self.status_monitor = status_monitor
        self.executor = executor
        self.decider = decider
    
    def on_position(self, position):
        my_pos = None
        his_pos = None
        cur_positions = self.executor.get_positions()
        price = self.executor.get_price("BTCUSDT")
        self.status_monitor.increment_counter()
        self.status_monitor.update_price(price)
        if len(cur_positions) > 0:
            self.status_monitor.report_my_position(cur_positions[0])
            my_pos = cur_positions[0]
        else:
            self.status_monitor.report_my_position(None)

        if position.leverage > 0:
            self.status_monitor.report_his_position(position)
            his_pos = position
        else:
            self.status_monitor.report_his_position(None)
        
        new_pos = self.decider.decide_position(my_pos, his_pos)
        self.executor.exec_position(new_pos)
        