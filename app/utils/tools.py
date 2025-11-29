import base64
import httpx
import pdfplumber
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO, StringIO
from playwright.sync_api import sync_playwright
from langchain_core.tools import tool


@tool
def scrape_js(url: str, selector: str = None): #type:ignore
    """Scrape a website using JS rendering. Returns text or full HTML.
    Args:
        url: The URL to scrape
        selector: Optional CSS selector to extract specific content
    """
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(url, wait_until="networkidle")
        content = (
            page.locator(selector).inner_text()
            if selector else page.content()
        )

        browser.close()
        return content

@tool
def fetch_api(url: str, headers: dict = None, params: dict = None): #type:ignore
    """Fetch JSON or text from an API.
    
    Args:
        url: The API endpoint URL
        headers: Optional HTTP headers
        params: Optional query parameters
    """
    r = httpx.get(url, headers=headers, params=params, timeout=30)
    try:
        return r.json()
    except Exception:
        return r.text


@tool
def extract_pdf(base64_pdf: str):
    """Extract text from a base64-encoded PDF.
    Args:
        base64_pdf: Base64 encoded PDF content
    """
    pdf_bytes = base64.b64decode(base64_pdf)
    with open("temp.pdf", "wb") as f:
        f.write(pdf_bytes)

    text = ""
    with pdfplumber.open("temp.pdf") as pdf:
        for page in pdf.pages:
            t = page.extract_text()
            if t:
                text += t + "\n"
    return text


@tool
def clean_data(csv_text: str):
    """Clean CSV data by dropping NaN values. Returns records as list of dicts.
    
    Args:
        csv_text: CSV content as string
    """
    df = pd.read_csv(StringIO(csv_text))
    df = df.dropna()
    return df.to_dict(orient="records")


@tool
def analyze_data(csv_text: str, op: str):
    """Analyze CSV data with basic operations.
    
    Args:
        csv_text: CSV content as string
        op: Operation to perform ('mean' or 'sum')
    """
    df = pd.read_csv(StringIO(csv_text))
    if op == "mean":
        return df.mean(numeric_only=True).to_dict()
    if op == "sum":
        return df.sum(numeric_only=True).to_dict()
    return {"error": "Invalid operation. Use 'mean' or 'sum'"}


@tool
def plot_data(csv_text: str, x: str, y: str):
    """Create a plot from CSV data and return as base64 PNG.
    
    Args:
        csv_text: CSV content as string
        x: Column name for x-axis
        y: Column name for y-axis
    """
    df = pd.read_csv(StringIO(csv_text))

    fig, ax = plt.subplots()
    df.plot(x=x, y=y, ax=ax)

    buf = BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)

    return base64.b64encode(buf.read()).decode()


@tool
def submit_answer(next_url: str, answer: str):
    """Submit the final quiz answer in structured format.
    This tool should be called after solving the quiz to return the result.
    
    Args:
        next_url: The URL to submit the answer to
        answer: The answer to the quiz
    """
    return {
        "next_url": next_url,
        "answer": answer
    }

tools = [
    scrape_js,
    fetch_api,
    extract_pdf,
    clean_data,
    analyze_data,
    plot_data,
    submit_answer,
]