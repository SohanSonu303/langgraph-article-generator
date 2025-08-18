# LangGraph Financial Analyst Agent

This project implements a sophisticated, autonomous financial analyst agent using Python and the LangGraph framework. Designed for efficiency and cost-effectiveness, the agent takes a stock ticker, performs comprehensive research, and generates a high-quality summary report.

The agent's key innovation is a **two-stage data processing pipeline** that drastically reduces operational costs and improves performance. It first uses a cost-effective model (`gpt-4o-mini`) to consolidate noisy web data before passing the clean, relevant information to a final summarizer. This is combined with a self-correction loop, where a critique agent reviews the summary and sends it back for revision, ensuring a polished and accurate final output.

## The Core Problem: The High Cost of Raw Data

Sending large volumes of raw, unstructured web content directly to a powerful LLM like GPT-4 is extremely expensive and inefficient. Scraped web pages are filled with boilerplate text, ads, navigation menus, and irrelevant information. This "noise" inflates the token count, increases API costs, and can distract the LLM, leading to lower-quality summaries.

This agent is specifically designed to solve this problem.

## Key Features

-   **Two-Stage Data Processing for Cost Optimization**: The agent first scrapes raw data and then uses a fast, inexpensive `gpt-4o-mini` node to consolidate it. This "consolidator" extracts only the key facts and figures, acting as an intelligent filter. This dramatically reduces the token load for the final, more complex analysis step.
-   **Powered by GPT-4o-mini**: The agent exclusively uses `gpt-4o-mini` for all its intelligence. This model was chosen for its excellent balance of high performance, speed, and industry-leading low cost, making it the perfect engine for both data extraction and final summary generation.
-   **Autonomous Workflow**: Built on LangGraph, the agent follows a robust, stateful graph of operations from data gathering to final report generation.
-   **Multi-Tool Integration**: Utilizes several tools to gather information:
    -   `yfinance`: Fetches historical stock price data.
    -   `Tavily AI`: Performs optimized web searches to find relevant news articles.
    -   `requests` & `BeautifulSoup`: Scrapes the content from the discovered news articles.
-   **Quantitative & Qualitative Analysis**: Automatically calculates key technical indicators (SMA, EMA, RSI, MACD) and synthesizes them with the consolidated news content for a holistic view.
-   **Self-Correction Loop**: A "summarizer" agent creates a report, which is then passed to a "critique" agent. If the critique agent finds flaws, the summary is revised. This loop is limited to a maximum of two revisions to ensure quality without getting stuck.
-   **Robust Scraping**: Implements a `User-Agent` header to mimic a real browser, bypassing basic anti-scraping measures and increasing the success rate of web scraping.

## How It Works: The Agent's Workflow

The agent operates as a state machine. The new `consolidate_data` node is the key to its efficiency.

```mermaid
graph TD
    A[get_stock_data] --> B[research];
    B --> C[scrape];
    C --> D[consolidate_data];
    D --> E[perform_technical_analysis];
    E --> F[summarize];
    F --> G{perform_critique};
    G -- "Critique Exists & Revisions < 2" --> F;
    G -- "Summary is Good OR Max Revisions Reached" --> H[generate_article];
    H --> I([END]);

```1.  **Get Stock Data**: Fetches 3 months of historical data for the ticker.
2.  **Research**: Uses Tavily to find recent, relevant news articles.
3.  **Scrape**: Scrapes the raw, unstructured text from the found URLs.
4.  **Consolidate Data (Key Step)**: Passes the noisy scraped text to `gpt-4o-mini` with a specific prompt to extract only the essential facts, figures, and analysis, discarding all irrelevant content.
5.  **Perform Technical Analysis**: Calculates SMA, EMA, RSI, and MACD from the stock data.
6.  **Summarize**: The main summarizer node receives the **clean, consolidated text** and the technical analysis. It uses `gpt-4o-mini` to synthesize this high-signal information into a coherent report.
7.  **Critique & Loop**: The generated summary is reviewed. If it needs improvement and the revision limit has not been met, it is sent back to the `summarize` node with feedback.
8.  **Generate Article**: The final, approved summary is prepared as the end product.

## Setup and Installation

1.  **Clone the Repository**
    ```sh
    git clone https://github.com/your-username/langgraph-financial-agent.git
    cd langgraph-financial-agent
    ```

2.  **Create and Activate a Virtual Environment**
    This project is stable on Python 3.11.
    ```sh
    # Create the virtual environment
    py -3.11 -m venv .venv

    # Activate it (on Windows)
    .\.venv\Scripts\Activate.ps1
    ```

3.  **Install Dependencies**
    A `requirements.txt` file is provided with known-good versions of all libraries.
    ```sh
    pip install -r requirements.txt
    ```

4.  **Set Up Environment Variables**
    Create a file named `.env` in the root of the project directory and add your API keys.
    ```.env
    OPENAI_API_KEY="sk-..."
    TAVILY_API_KEY="tvly-..."
    ```

## How to Run

Ensure your virtual environment is activated, then run the `main.py` script:
```sh
python main.py
```
The script will then prompt you to enter a stock ticker (e.g., `GOOGL`, `MSFT`, `SBIN.NS`).