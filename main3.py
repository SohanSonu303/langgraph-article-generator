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
    scraped_data: str
    concise_data: str
    technical_analysis: str
    summary: str
    critique: str
    article: str
    revision_number: int

# --- 2. Define the Nodes ---
def get_stock_data(state: GraphState) -> GraphState:
    """ Fetches historical stock data. """
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
    Performs web research and retrieves pre-scraped content using Tavily.
    """
    print("---RESEARCHING & GATHERING CONTENT---")
    query = f"Latest news and stock analysis for {state['ticker']}"
    tavily = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])
    results = tavily.search(query=query, max_results=3, include_raw_content=True)
    
    print("\n---TAVILY SEARCH RESULTS---")
    print(results)
    
    all_scraped_data = ""
    # FIX: Make the loop robust to handle cases where 'raw_content' is None
    for result in results['results']:
        # Get the raw content for the current result
        content = result.get('raw_content')
        # Check if the content is not None before appending
        if content:
            all_scraped_data += content + "\n"
        
    print(f"---CONTENT GATHERED---")
    return {"scraped_data": all_scraped_data}

def concise_data(state: GraphState) -> GraphState:
    """
    Uses gpt-4o-mini to consolidate the scraped data into a concise format.
    """
    print("---CONSOLIDATING DATA---")
    scraped_data = state["scraped_data"]
    if not scraped_data.strip():
        print("---NO DATA TO CONSOLIDATE---")
        return {"concise_data": "No news content was found."}

    prompt = ChatPromptTemplate.from_template(
        """You are a data extraction specialist.
        Given the following raw text from financial news websites, extract only the key facts, figures, and direct analysis related to the stock.
        Ignore boilerplate text, ads, navigation links, and irrelevant information.
        If the text is empty or contains no relevant information, state that 'No relevant news content was found.'

        Raw Text:
        {scraped_data}
        """
    )
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)
    output_parser = StrOutputParser()
    chain = prompt | llm | output_parser
    concise_summary = chain.invoke({"scraped_data": scraped_data})
    
    print("\n---CONSOLIDATED DATA---")
    print(concise_summary)
    
    return {"concise_data": concise_summary}

def technical_analysis(state: GraphState) -> GraphState:
    """ Performs technical analysis. """
    print("---PERFORMING TECHNICAL ANALYSIS---")
    stock_data = state["stock_data"]
    if stock_data.empty:
        return {"technical_analysis": "No stock data available for technical analysis."}
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
    Generates a final summary using the consolidated data and technical analysis.
    """
    print("---GENERATING FINAL SUMMARY---")
    consolidated_data = state["concise_data"]
    technical_analysis = state["technical_analysis"]
    query = state["query"]
    critique = state.get("critique")
    revision_number = state["revision_number"]

    prompt_text = """You are an expert financial analyst.
    Your task is to write a comprehensive and well-structured summary for the stock: '{query}'.
    Synthesize the information from the provided CONSOLIDATED news summary and the quantitative technical indicators to create a holistic final report.

    Consolidated News Summary:
    {consolidated_data}

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
        "query": query, "consolidated_data": consolidated_data,
        "technical_analysis": technical_analysis, "critique": critique
    })
    print("---FINAL SUMMARY GENERATED---")
    return {"summary": summary, "revision_number": revision_number + 1}

def critique(state: GraphState) -> GraphState:
    """ Critiques the summary. """
    print("---CRITIQUING SUMMARY---")
    summary = state["summary"]
    prompt = ChatPromptTemplate.from_template(
        """You are an expert financial editor. Critique the following summary. 
        Be specific and provide actionable feedback. If the summary is good, respond with 'None'.
        Summary: {summary}"""
    )
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
    output_parser = StrOutputParser()
    chain = prompt | llm | output_parser
    critique_text = chain.invoke({"summary": summary})
    print("---CRITIQUED GENERATED---")
    if "none" in critique_text.lower() and len(critique_text) < 10:
        return {"critique": None}
    else:
        return {"critique": critique_text}

# --- 3. Build the Graph ---
def decide_to_continue(state: GraphState) -> str:
    """ Decides the next step with a revision limit. """
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
workflow.add_node("consolidate_data", concise_data)
workflow.add_node("perform_technical_analysis", technical_analysis)
workflow.add_node("summarize", summarize)
workflow.add_node("perform_critique", critique)
workflow.add_node("generate_article", lambda state: {"article": state["summary"]})

# Set the entry point of the graph
workflow.set_entry_point("get_stock_data")

# Add edges to define the flow
workflow.add_edge("get_stock_data", "research")
workflow.add_edge("research", "consolidate_data")
workflow.add_edge("consolidate_data", "perform_technical_analysis")
workflow.add_edge("perform_technical_analysis", "summarize")
workflow.add_edge("summarize", "perform_critique")
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
    inputs = { "query": user_query, "revision_number": 0 }
    final_state = app.invoke(inputs)
    print("\n\n--- FINAL ARTICLE ---")
    print(final_state["article"])