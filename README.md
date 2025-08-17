# LangGraph Financial Analyst Agent

This project implements an autonomous financial analyst agent using Python and the LangGraph framework. The agent takes a stock ticker as input, performs comprehensive research by gathering both quantitative (stock data) and qualitative (news articles) information, and generates a cohesive summary report. A key feature of this agent is its self-correction loop, where a "critique" agent reviews the summary and sends it back for revision, ensuring a high-quality final output.

## Key Features

-   **Autonomous Workflow**: Built on LangGraph, the agent follows a predefined graph of operations from data gathering to final report generation.
-   **Multi-Tool Integration**: Utilizes several tools to gather information:
    -   `yfinance`: Fetches historical stock price data.
    -   `Tavily AI`: Performs optimized web searches to find relevant news articles.
    -   `requests` & `BeautifulSoup`: Scrapes the content from the discovered news articles.
-   **Quantitative Analysis**: Automatically calculates key technical indicators like SMA, EMA, RSI, and MACD using the `pandas-ta` library.
-   **Qualitative Analysis**: Scrapes and processes the text from recent news to understand market sentiment and current events.
-   **Self-Correction Loop**: A "summarizer" agent creates a report, which is then passed to a "critique" agent. If the critique agent finds flaws, it provides feedback, and the summary is revised. This loop is limited to a maximum of two revisions to prevent infinite cycles.
-   **Robust Scraping**: Implements a `User-Agent` header to mimic a real browser, increasing the success rate of web scraping on modern websites.

## How It Works: The Agent's Workflow

The agent operates as a state machine, where each step (or "node") passes its results to the next. The core logic is as follows:

```mermaid
graph TD
    A[get_stock_data] --> B[research];
    B --> C[scrape];
    C --> D[perform_technical_analysis];
    D --> E[summarize];
    E --> F{perform_critique};
    F -- "Critique Exists & Revisions < 2" --> E;
    F -- "Summary is Good OR Max Revisions Reached" --> G[generate_article];
    G --> H([END]);


