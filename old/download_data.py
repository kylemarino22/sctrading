import time
import requests
from datetime import datetime

import threading
import sctrading.raw_data.file_io as file_io
from typing import List





class PolygonDataDownloader:
    
    def __init__ (self):

        self.api_key = ""
        self.base_url = "https://api.polygon.io"

        self._metadata_lock = threading.Lock()

        
    def download(full_download=False):
        
        
        
        
        
        return

    def get_tickers(self, update=True):

        if not update:
            # Load active tickers from file
            active_tickers = file_io.get_tickers(active=True)
            delisted_tickers = file_io.get_tickers(active=False)
            return active_tickers, delisted_tickers        
        

        today = datetime.today().strftime("%Y%m%d")
        # Get update date stored in active_tickers.json
        last_active_update = file_io.get_last_ticker_update(active=True)

        if today != last_active_update:
            # Download active tickers
            active_tickers = self.download_tickers(active=True)
            file_io.store_tickers(active_tickers, today, active=True)


        # Get update date stored in delisted_tickers.json
        last_delisted_update = file_io.get_last_ticker_update(active=False)
        if today != last_delisted_update:
            # Download delisted tickers
            delisted_tickers = self.download_tickers(active=False)
            file_io.store_tickers(delisted_tickers, today, active=False)

            
        return active_tickers, delisted_tickers
        

    def download_tickers(self, active: bool) -> List:

        symbols = []
        url = f"{self.base_url}/v3/reference/tickers"
        params = {
            "market": "stocks",
            "active": "true" if active else "false",
            "limit": 1000,
            "apiKey": self.api_key
        }

        page = 1
        while True:
            resp = requests.get(url, params=params).json()
            batch = resp.get("results", [])
            print(f"[{'A' if active else 'D'}:{page}] Retrieved {len(batch)} tickers")
            symbols.extend([r["ticker"] for r in batch if r.get("ticker")])

            nxt = resp.get("next_url")
            if not nxt:
                break
            if "apiKey=" not in nxt:
                sep = "&" if "?" in nxt else "?"
                nxt = f"{nxt}{sep}apiKey={self.api_key}"
            url = nxt
            params = None
            page += 1
            time.sleep(0.1)

        return symbols


    def full_download():
        pass

    
    def update():
        
        pass

    
    

        
        
if __name__ == "__main__":

    downloader = PolygonDataDownloader()


    
