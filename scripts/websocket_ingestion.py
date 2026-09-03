"""
Real-Time Binance WebSocket Ingestion Client with Kafka Streaming
==================================================================
Connects via asyncio/websockets to Binance live trade and depth streams,
handles network disconnects with exponential backoff, and publishes trades
to Apache Kafka (with local buffer fallback).
"""

import os
import sys
import json
import time
import asyncio
import argparse
import random
from typing import Dict, Any, List, Optional
import websockets

# Ensure project root in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from scripts.utils import setup_logger, load_config, get_project_root

# Try importing KafkaProducer with graceful fallback
try:
    from kafka import KafkaProducer
    KAFKA_AVAILABLE = True
except ImportError:
    try:
        from kafka_python_ng import KafkaProducer
        KAFKA_AVAILABLE = True
    except ImportError:
        KAFKA_AVAILABLE = False


class WebSocketTradeIngestor:
    """Async WebSocket streaming client for Binance trade channels."""

    def __init__(
        self,
        symbols: List[str],
        kafka_bootstrap: str = "localhost:9092",
        raw_topic: str = "crypto-raw-trades",
        enable_kafka: bool = True,
        logger: Optional[Any] = None
    ):
        self.symbols = [s.lower().strip() for s in symbols]
        self.kafka_bootstrap = kafka_bootstrap
        self.raw_topic = raw_topic
        self.enable_kafka = enable_kafka and KAFKA_AVAILABLE
        self.logger = logger or setup_logger("ws_ingestor")
        self.running = False
        self.message_count = 0
        self.producer = None
        self.local_buffer = []

        self._init_kafka()

    def _init_kafka(self):
        """Initializes Kafka producer or defaults to in-memory fallback buffer."""
        if self.enable_kafka:
            try:
                self.producer = KafkaProducer(
                    bootstrap_servers=self.kafka_bootstrap.split(","),
                    value_serializer=lambda v: json.dumps(v).encode("utf-8"),
                    key_serializer=lambda k: k.encode("utf-8") if k else None,
                    retries=3,
                    linger_ms=10,
                    batch_size=16384,
                    request_timeout_ms=5000
                )
                self.logger.info(f"Connected to Kafka broker at {self.kafka_bootstrap}. Topic: {self.raw_topic}")
            except Exception as e:
                self.logger.warning(f"Kafka unavailable ({e}). Using local high-speed memory buffer.")
                self.producer = None
        else:
            self.logger.info("Kafka disabled or library not present. Ingesting to local memory buffer.")

    def _publish_event(self, symbol: str, event_data: Dict[str, Any]):
        """Publishes parsed trade event to Kafka or memory buffer."""
        self.message_count += 1
        if self.producer:
            try:
                self.producer.send(self.raw_topic, key=symbol, value=event_data)
            except Exception as e:
                self.logger.error(f"Kafka send failed: {e}")
                self.local_buffer.append(event_data)
        else:
            self.local_buffer.append(event_data)
            if len(self.local_buffer) > 10000:
                self.local_buffer.pop(0)  # Evict oldest to keep memory bounded

        if self.message_count % 50 == 0:
            price = event_data.get("p", "N/A")
            qty = event_data.get("q", "N/A")
            self.logger.info(f"Stream [{symbol.upper()}]: Ingested {self.message_count} trades. Last: Price={price}, Qty={qty}")

    def build_stream_url(self) -> str:
        """Constructs combined stream URL for multi-asset monitoring."""
        if len(self.symbols) == 1:
            return f"wss://stream.binance.com:9443/ws/{self.symbols[0]}@trade"
        streams = "/".join([f"{s}@trade" for s in self.symbols])
        return f"wss://stream.binance.com:9443/stream?streams={streams}"

    async def stream_trades(self, max_duration_sec: Optional[int] = None):
        """
        Connects to Binance WebSocket stream with automatic reconnect and exponential backoff.
        """
        self.running = True
        url = self.build_stream_url()
        start_time = time.time()
        backoff = 1.0

        self.logger.info(f"Initiating WebSocket connection to: {url}")

        while self.running:
            if max_duration_sec and (time.time() - start_time) >= max_duration_sec:
                self.logger.info(f"Specified stream duration ({max_duration_sec}s) reached. Stopping.")
                break

            try:
                async with websockets.connect(
                    url,
                    ping_interval=20,
                    ping_timeout=15,
                    close_timeout=10,
                    max_size=2**22
                ) as ws:
                    self.logger.info("WebSocket handshake successful. Receiving real-time trade packets...")
                    backoff = 1.0  # Reset backoff upon successful connection

                    while self.running:
                        if max_duration_sec and (time.time() - start_time) >= max_duration_sec:
                            break

                        try:
                            msg_text = await asyncio.wait_for(ws.recv(), timeout=30.0)
                            msg = json.loads(msg_text)

                            # Handle single stream vs combined streams payload format
                            if "data" in msg:
                                trade_payload = msg["data"]
                                symbol = trade_payload.get("s", "UNKNOWN")
                            else:
                                trade_payload = msg
                                symbol = trade_payload.get("s", self.symbols[0].upper())

                            # Normalize payload
                            trade_payload["ingest_time_ms"] = int(time.time() * 1000)
                            self._publish_event(symbol, trade_payload)

                        except asyncio.TimeoutError:
                            self.logger.debug("Keepalive heartbeat ping...")
                            await ws.ping()

            except (websockets.ConnectionClosed, websockets.WebSocketException, OSError) as e:
                self.logger.warning(f"WebSocket disconnected ({e}). Reconnecting in {backoff:.1f}s...")
                await asyncio.sleep(backoff)
                backoff = min(backoff * 1.8 + random.uniform(0.1, 0.5), 60.0)
            except Exception as e:
                self.logger.error(f"Unexpected stream error: {e}. Backing off {backoff:.1f}s...")
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2.0, 60.0)

        self.stop()

    def stop(self):
        """Stops the streaming client and closes connections cleanly."""
        self.running = False
        if self.producer:
            try:
                self.producer.flush(timeout=5)
                self.producer.close(timeout=5)
                self.logger.info("Kafka producer flushed and shut down cleanly.")
            except Exception as e:
                self.logger.error(f"Error closing Kafka producer: {e}")
        self.logger.info(f"WebSocket Ingestion finished. Total trades ingested: {self.message_count}")


def main():
    parser = argparse.ArgumentParser(description="Real-time Binance WebSocket Ingestor to Kafka.")
    parser.add_argument("--symbols", nargs="+", default=["btcusdt", "ethusdt", "solusdt"], help="Trading pair symbols")
    parser.add_argument("--duration", type=int, default=30, help="Stream duration in seconds (0 for indefinite)")
    parser.add_argument("--kafka", type=str, default="localhost:9092", help="Kafka broker address")
    parser.add_argument("--no-kafka", action="store_true", help="Disable Kafka publishing and use local buffer")
    args = parser.parse_args()

    config = load_config()
    symbols = args.symbols or config.get("ingestion", {}).get("binance", {}).get("symbols", ["btcusdt"])
    duration = None if args.duration <= 0 else args.duration

    ingestor = WebSocketTradeIngestor(
        symbols=symbols,
        kafka_bootstrap=args.kafka,
        enable_kafka=not args.no_kafka
    )

    try:
        asyncio.run(ingestor.stream_trades(max_duration_sec=duration))
    except KeyboardInterrupt:
        print("\nShutdown signal received.")
        ingestor.stop()


if __name__ == "__main__":
    main()
