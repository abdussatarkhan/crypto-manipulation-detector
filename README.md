# Crypto Manipulation & Wash-Trading Anomaly Detection Engine

[![CI](https://github.com/abdussatarkhan/crypto-manipulation-detector/actions/workflows/ci.yml/badge.svg)](https://github.com/abdussatarkhan/crypto-manipulation-detector/actions)
[![Python](https://img.shields.io/badge/Python-Forensics-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/) [![Anomaly Detection](https://img.shields.io/badge/ML-Isolation_Forest-FFA800?style=for-the-badge&logo=scikitlearn&logoColor=white)](https://scikit-learn.org/) [![Docker](https://img.shields.io/badge/Docker-Multi--Container-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://www.docker.com/)
[![Author](https://img.shields.io/badge/Author-Abdussatar-E50914?style=for-the-badge&logo=github&logoColor=white)](https://github.com/abdussatarkhan)

> **A quantitative forensics and anomaly detection pipeline identifying cryptocurrency wash-trading, spoofing, and cyclical volume manipulation using Benford's Law analysis, network graph centrality, and unsupervised machine learning.**

---

## 🏛️ System Architecture

```mermaid
graph TD
    TradeData[Real-Time L2 Orderbook & Trade Feed] --> Benford[Benford First-Digit Statistical Test]
    TradeData --> Graph[NetworkX Cycle & Wash Trading Graph Analysis]
    Benford --> Ensemble[Composite Manipulation Scoring Engine]
    Graph --> Ensemble
    Ensemble --> Alert[Regulatory Alert & Compliance Report]
```

---

## 🌟 Key Features & Capabilities

- **Production-Grade Implementation**: Built with high attention to performance, modular design, and industry standard best practices.
- **Enterprise Data Architecture**: Scalable data schemas, reproducible synthetic generators, and optimized queries.
- **Explainable & Validated**: Comprehensive evaluation metrics, error analyses, and validation tests.
- **Comprehensive Tech Stack**: `Python` `Pandas` `NetworkX` `Scikit-Learn` `Benford's Law` `Docker`.

---

## 📊 Visual Preview & Analysis

<div align="center">

![crypto-manipulation-detector preview](images/benfords_law_anomaly.png)

</div>

---

## 🚀 Quickstart & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/abdussatarkhan/crypto-manipulation-detector.git
cd crypto-manipulation-detector
```

### 2. Environment Setup
```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate

# Install dependencies (if requirements.txt exists)
pip install -r requirements.txt
```

---

## 🗺️ Roadmap & Upcoming Features

- [x] Benford's Law leading digit anomaly detection
- [x] Network graph wash-trading cycle detection
- [ ] Live Binance / Coinbase WebSocket real-time orderbook stream
- [ ] PyVis / D3.js interactive network graph renderer
- [ ] Automated regulatory compliance alert webhook (Discord/Slack)

---

## 👨‍💻 Author & Profile

Built and maintained by **Abdussatar** ([@abdussatarkhan](https://github.com/abdussatarkhan)).  
For technical discussions, collaboration, or queries, feel free to reach out via [LinkedIn](https://www.linkedin.com/in/abdus-satar-5150813b5/) or [GitHub](https://github.com/abdussatarkhan).

---

## 📜 License

This project is licensed under the **MIT License** — see the LICENSE file for details.