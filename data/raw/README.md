# Data Sources & Ingestion Guide

This document outlines the external APIs, data schemas, rate limits, and authentication protocols used in the **Crypto Wash-Trade & Market Manipulation Detector**.

---

## 1. Binance Market Data API

### A. REST API (Historical Trades & Klines)
- **Base Endpoint**: `https://api.binance.com/api/v3`
- **Authentication**: Public endpoints require no API key; private account monitoring requires HMAC-SHA256 signature and `X-MBX-APIKEY` header.
- **Rate Limits**: 1,200 request weight per minute per IP. Exceeding triggers HTTP `429 Too Many Requests`.

#### Key Endpoints
1. **Aggregated Trades (`/api/v3/aggTrades`)**:
   - Compresses multiple fills from the same taker order into single aggregate events.
   - Parameters: `symbol` (e.g., `BTCUSDT`), `fromId`, `startTime`, `endTime`, `limit` (default 500, max 1000).
   - Payload Format:
     ```json
     [
       {
         "a": 26129,         // Aggregate tradeId
         "p": "0.01633102",  // Price
         "q": "4.70443515",  // Quantity
         "f": 27781,         // First tradeId
         "l": 27781,         // Last tradeId
         "T": 1498793709153, // Timestamp (ms)
         "m": true,          // Was the buyer the maker?
         "M": true           // Was the trade the best price match?
       }
     ]
     ```

2. **Historical Kline / Candlesticks (`/api/v3/klines`)**:
   - Parameters: `symbol`, `interval` (`1m`, `5m`, `1h`, etc.), `startTime`, `endTime`, `limit` (max 1000).
   - Payload Format:
     ```json
     [
       [
         1499040000000,      // Open time
         "0.01634790",       // Open
         "0.80000000",       // High
         "0.01575800",       // Low
         "0.01577100",       // Close
         "148976.11427815",  // Volume
         1499644799999,      // Close time
         "2434.19055334",    // Quote asset volume
         308,                // Number of trades
         "1756.87402397",    // Taker buy base asset volume
         "28.46694368",      // Taker buy quote asset volume
         "17928899.62484339" // Ignore
       ]
     ]
     ```

### B. WebSocket Streams (Real-Time Ingestion)
- **Base WebSocket URL**: `wss://stream.binance.com:9443/ws` or `/stream?streams=`
- **Individual Trade Stream**: `<symbol>@trade` (e.g., `btcusdt@trade`, `ethusdt@trade`)
- **Order Book Depth Stream**: `<symbol>@depth20@100ms` (for bid-ask spread and order-cancel monitoring)
- **Aggregated Trade Stream**: `<symbol>@aggTrade`
- **Format**:
  ```json
  {
    "e": "trade",
    "E": 1672531199000,
    "s": "BTCUSDT",
    "t": 123456789,
    "p": "16500.25",
    "q": "0.1500",
    "b": 882194,
    "a": 882205,
    "T": 1672531198995,
    "m": true,
    "M": true
  }
  ```

---

## 2. CoinGecko API (Market Context & Exchange Metadata)

- **Base Endpoint**: `https://api.coingecko.com/api/v3`
- **Rate Limit**: Free tier allows 10-30 calls/minute.
- **Usage**:
  - Exchange volume cross-verification (`/exchanges/{id}/tickers`)
  - Market capitalization and global volume benchmarking (`/coins/{id}/market_chart`)
  - Identifying unverified volume surges relative to aggregate market volume across 50+ exchanges.

---

## 3. Blockchain.com Explorer API (On-Chain Graph Analytics)

- **Base Endpoint**: `https://blockchain.info/rawaddr/$bitcoin_address` or `https://blockchain.info/rawtx/$tx_hash`
- **Usage**:
  - Pulling wallet transaction graphs, inputs, and outputs.
  - Tracking circular fund flows (e.g., Address A -> Address B -> Address C -> Address A).
  - Exchange deposit and withdrawal clustering to detect coordinated wash trading rings.
- **Data Model**:
  - **Nodes**: Addresses / Wallets (clustered by exchange hot/cold wallets vs. user wallets).
  - **Edges**: Transactions with timestamp, value (satoshis/BTC), and fee.

---

## 4. Local Data Storage & Processing Architecture

- **Raw Ingestion Directory**: `data/raw/` (excluded from git via `.gitignore`).
- **Data Formats**:
  - Raw WebSocket streams buffered into Apache Kafka topic: `binance-trades`.
  - Processed parquet partitions: `data/processed/trades_{date}.parquet`.
  - Transaction graph cache: `data/processed/network_graph_{timestamp}.gpickle` or `.graphml`.
- **Sample Generation Script**:
  Run `python scripts/data_collection_historical.py --symbol BTCUSDT --days 7` to populate raw datasets.
