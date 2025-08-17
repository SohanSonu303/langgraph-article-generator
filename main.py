import os
import pandas as pd
import yfinance as yf
import pandas_ta as ta
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from tavily import TavilyClient
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv


load_dotenv()
# --- 1. Define the State ---
class GraphState(TypedDict):
    """
    Represents the state of our graph.
    """
    query: str
    ticker: str
    stock_data: pd.DataFrame
    links: List[str]
    scraped_data: str
    technical_analysis: str
    summary: str
    critique: str
    article: str
    revision_number: int

# --- 2. Define the Nodes ---
def get_stock_data(state: GraphState) -> GraphState:
    """
    Fetches historical stock data.
    """
    print("---FETCHING STOCK DATA---")
    query = state["query"]
    ticker_symbol = query
    ticker = yf.Ticker(ticker_symbol)
    stock_data = ticker.history(period="3mo")

    if stock_data.empty:
        print(f"---NO DATA FOUND FOR TICKER: {ticker_symbol}---")
        return {"stock_data": pd.DataFrame(), "ticker": ticker_symbol}

    print(f"---DATA FETCHED FOR {ticker_symbol}---")
    return {"stock_data": stock_data, "ticker": ticker_symbol}


def research(state: GraphState) -> GraphState:
    """
    Performs web research.
    """
    print("---RESEARCHING NEWS---")
    query = f"Latest stock news and analysis for {state['ticker']}"
    tavily = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])
    results = tavily.search(query=query, max_results=3)

    print("\n---TAVILY SEARCH RESULTS---")
    print(results)

    links = [result["url"] for result in results["results"]]
    print(f"---FOUND {len(links)} LINKS---")
    return {"links": links}

def scrape(state: GraphState) -> GraphState:
    """
    Scrapes content from URLs using a User-Agent to appear as a browser.
    """
    print("---SCRAPING---")
    urls = state["links"]
    all_scraped_data = ""
    
    # NEW: Define a User-Agent header to mimic a web browser
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }

    for url in urls:
        try:
            # MODIFIED: Add the headers to the request
            response = requests.get(url, headers=headers, timeout=10)
            # Raise an exception if the request was not successful (e.g., 403 Forbidden)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # A slightly more robust way to get text, not just from <p> tags
            text_content = soup.body.get_text(separator='\n', strip=True)
            all_scraped_data += text_content + "\n\n"

        except Exception as e:
            print(f"Error scraping {url}: {e}")
            
    print("---SCRAPING COMPLETE---")
    
    print("\n---SCRAPED NEWS CONTENT---")
    print(all_scraped_data[:2000] + "..." if len(all_scraped_data) > 2000 else all_scraped_data)

    return {"scraped_data": all_scraped_data}

def technical_analysis(state: GraphState) -> GraphState:
    """
    Performs technical analysis.
    """
    print("---PERFORMING TECHNICAL ANALYSIS---")
    stock_data = state["stock_data"]
    if stock_data.empty:
        return {"technical_analysis": "No data available for technical analysis."}

    stock_data.ta.sma(length=20, append=True)
    stock_data.ta.ema(length=50, append=True)
    stock_data.ta.rsi(length=14, append=True)
    stock_data.ta.macd(fast=12, slow=26, signal=9, append=True)
    
    latest_data = stock_data.iloc[-1]
    
    analysis_str = f"""
    Latest Technical Indicators:
    - SMA (20): {latest_data['SMA_20']:.2f}
    - EMA (50): {latest_data['EMA_50']:.2f}
    - RSI (14): {latest_data['RSI_14']:.2f}
    - MACD: {latest_data['MACD_12_26_9']:.2f}
    - MACD Signal: {latest_data['MACDs_12_26_9']:.2f}
    - MACD Histogram: {latest_data['MACDh_12_26_9']:.2f}
    """
    print("---TECHNICAL ANALYSIS COMPLETE---")
    return {"technical_analysis": analysis_str}

def summarize(state: GraphState) -> GraphState:
    """
    Generates a summary and increments the revision counter.
    """
    print("---GENERATING SUMMARY---")
    scraped_data = state["scraped_data"]
    technical_analysis = state["technical_analysis"]
    query = state["query"]
    critique = state.get("critique")
    revision_number = state["revision_number"]

    prompt_text = """You are an expert financial analyst.
    Your task is to write a concise summary based on the provided news and technical analysis for the stock: '{query}'.
    Combine the insights from the scraped news content and the quantitative technical indicators to provide a holistic view.

    Scraped News Content:
    {scraped_data}

    Technical Analysis:
    {technical_analysis}
    """
    if critique:
        prompt_text += """
        You have received the following critique on your previous summary. 
        Please revise your summary to address these points:
        Critique: {critique}
        """

    prompt = ChatPromptTemplate.from_template(prompt_text)
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
    output_parser = StrOutputParser()
    chain = prompt | llm | output_parser

    summary = chain.invoke({
        "query": query, "scraped_data": scraped_data,
        "technical_analysis": technical_analysis, "critique": critique
    })
    print("---SUMMARY GENERATED---")
    return {"summary": summary, "revision_number": revision_number + 1}

def critique(state: GraphState) -> GraphState:
    """
    Critiques the summary.
    """
    print("---CRITIQUING SUMMARY---")
    summary = state["summary"]
    prompt = ChatPromptTemplate.from_template(
        """You are an expert financial editor.
        Your task is to critique the following summary. 
        Please be specific and provide actionable feedback. If the summary is good, respond with 'None'.

        Summary:
        {summary}
        """
    )
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
    output_parser = StrOutputParser()
    chain = prompt | llm | output_parser
    critique_text = chain.invoke({"summary": summary})
    print("---CRITIQUE GENERATED---")
    if "none" in critique_text.lower() and len(critique_text) < 10:
        return {"critique": None}
    else:
        return {"critique": critique_text}

# --- 3. Build the Graph ---
def decide_to_continue(state: GraphState) -> str:
    """
    Decides the next step, now with a revision limit.
    """
    revision_number = state["revision_number"]
    if revision_number >= 2:
        print(f"---MAX REVISIONS REACHED ({revision_number}), FINALIZING---")
        return "generate_article"

    if state["critique"]:
        print("---CRITIQUE RECEIVED, REVISING SUMMARY---")
        return "summarize"
    else:
        print("---SUMMARY APPROVED, FINALIZING---")
        return "generate_article"

workflow = StateGraph(GraphState)

# Add the nodes to the graph
workflow.add_node("get_stock_data", get_stock_data)
workflow.add_node("research", research)
workflow.add_node("scrape", scrape)
workflow.add_node("perform_technical_analysis", technical_analysis)
workflow.add_node("summarize", summarize)
workflow.add_node("perform_critique", critique)
workflow.add_node("generate_article", lambda state: {"article": state["summary"]})

# Set the entry point of the graph
workflow.set_entry_point("get_stock_data")

# Add edges to define the flow
workflow.add_edge("get_stock_data", "research")
workflow.add_edge("research", "scrape")
workflow.add_edge("scrape", "perform_technical_analysis")
workflow.add_edge("perform_technical_analysis", "summarize")
workflow.add_edge("summarize", "perform_critique")

# Add the conditional edge
workflow.add_conditional_edges(
    "perform_critique",
    decide_to_continue,
    {"summarize": "summarize", "generate_article": "generate_article"}
)
workflow.add_edge("generate_article", END)

# Compile the graph into a runnable application
app = workflow.compile()

if __name__ == "__main__":
    user_query = input("Enter the stock ticker for analysis (e.g., BHARTIARTL.NS): ")
    
    inputs = {
        "query": user_query,
        "revision_number": 0
    }
    
    final_state = app.invoke(inputs)

    print("\n\n--- FINAL ARTICLE ---")
    print(final_state["article"])