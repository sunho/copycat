from io import BytesIO
from rapidocr_onnxruntime import RapidOCR
from dataclasses import dataclass
import re

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

class Engine:
    def __init__(self, api_key, api_secret):
        self.trader = Trader(api_key, api_secret)
        self.my_position = None
        self.his_position = None

    def start(self, url):
        hls_url = get_live_stream_hls_url(url)
        cap = cv2.VideoCapture(hls_url)

        if not cap.isOpened():
            print("Error: Cannot open the video stream.")
            return
        

        layout = Layout()
        layout.split_column(
            Layout(name="upper"),
            Layout(name="lower")
        )

        layout["upper"].update(Panel(Text("Live Trading Engine Status", style="bold white on blue"), title="Header"))
        cnt = 0
        with Live(layout, refresh_per_second=1):
            while True:
                table = Table(title="Trading Status", expand=True)
                table.add_column("Symbol", justify="left", style="cyan", no_wrap=True)
                table.add_column("Side", justify="left", style="magenta")
                table.add_column("Position", justify="right", style="green")
                table.add_column("Avg Price", justify="right", style="yellow")
                table.add_column("Leverage", justify="right", style="blue")
                table.add_column("Owner", justify="right", style="blue")
                ret, frame = cap.read()
                if not ret:
                    print("Failed to grab frame")
                    break

                frame = self.crop_frame(frame)
                if cnt % 1200 == 0:
                    print("Trigger feed frame")
                    thread = threading.Thread(target=self.feed_frame, args=(frame,))
                    thread.daemon = True
                    thread.start()
                cnt += 1
                if self.my_position:
                    pos = self.my_position
                    side = "SHORT" if pos.is_short else "LONG"
                    table.add_row(pos.symbol, side, f"{pos.position}", f"{pos.avg_price}", f"{pos.leverage}x", "MY")
                if self.his_position:
                    pos = self.his_position
                    side = "SHORT" if pos.is_short else "LONG"
                    table.add_row(pos.symbol, side, f"{pos.position}", f"{pos.avg_price}", f"{pos.leverage}x", "HIS")

                layout["lower"].update(Panel(table, title="Positions"))

                cv2.imshow('Live Frame', frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        cap.release()
        cv2.destroyAllWindows()

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
    
    def feed_frame(self, frame):
        buf = self.frame_to_png_buffer(frame)
        extracter = TradeStateExtracter()
        his_position = extracter.extract_state(buf)
        print("Detected:", his_position)
        cur_position = self.trader.get_isolated_positions()
        self.his_position = his_position if his_position.position > 0 else None
        self.my_position = cur_position[0] if len(cur_position) > 0 else None
        if his_position.leverage == 0:
            self.trader.close_all()
        else:
            if len(cur_position) == 0:
                his_position.position *= RATIO
                self.trader.create_isolated_order(his_position)
            else:
                if his_position.is_short != cur_position[0].is_short:
                    self.trader.close_all()
                    his_position.position *= RATIO
                    self.trader.create_isolated_order(his_position)

class VideoFeed:
    def get_video(self):
        pass




@dataclass
class Position:
    symbol: str
    is_short: bool
    position: int
    avg_price: float
    leverage: int 

class Trader:
    def __init__(self, api_key, api_secret):
        self.client = Client(api_key, api_secret)

    def get_isolated_positions(self) -> list[Position]:
        positions = []
        try:
            # Fetch futures account details
            account_info = self.client.futures_account()
            
            # Loop through positions and filter non-zero isolated positions
            for position_data in account_info["positions"]:
                # Check if the position is isolated and has a non-zero amount
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
            print("An error occurred:", e)
        
        return positions

    def create_isolated_order(self, position: Position):
        symbol = "BTCUSDT"
        side = position.is_short and SIDE_SELL or SIDE_BUY
        qty = position.position / position.leverage
        unit_qty_tick = 0.001
        qty = round(qty / unit_qty_tick) * unit_qty_tick
        qty = max(qty,0.002)

        try:
            self.client.futures_change_margin_type(symbol=symbol, marginType="ISOLATED")
        except Exception:
            pass

        try:
            self.client.futures_change_leverage(symbol=symbol, leverage=position.leverage)
        except Exception:
            pass

        order = self.client.futures_create_order(
            symbol=symbol,
            side=side,
            type="MARKET",
            quantity=qty,
        )
        print("Order placed successfully:", order)
    
    def close_all(self):
        positions = self.get_isolated_positions()
        for position in positions:
            position.is_short = not position.is_short
            self.create_isolated_order(position)
    
@dataclass
class TradeState:
    positions: list[Position]

class TradeStateExtracter:
    def extract_state(self, img: bytes) -> Position:
        position = Position("BTCUSDT", True, 0, 0, 0)
        engine = RapidOCR()
        result, elapse = engine(img)
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
        
