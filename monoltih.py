from fastapi import FastAPI,Request,HTTPException,Response,BackgroundTasks
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional
import os
from dotenv import load_dotenv
import httpx
from typing import Annotated, TypedDict
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
import base64
import pdfplumber
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO, StringIO
from playwright.sync_api import sync_playwright
from langchain_core.tools import tool

load_dotenv()

app = FastAPI()
SECRET = os.getenv("secret")
KEY=os.getenv("api_key")


class req(BaseModel):
    url: str

class Quiz(BaseModel):
    url: str
    email:str
    secret:str

class reponse(BaseModel):
    correct:str
    url:str | None
    reason: str | None

class QuizResponse(BaseModel):
    correct: bool
    url: Optional[str] = None
    reason: Optional[str] = None

    model_config = {
        "extra": "ignore"
    }

class Test(BaseModel):
    submit_url:str="https://tds-llm-analysis.s-anand.net/submit"
    email: str = "your email"
    secret: str = "your secret"
    url: str = "https://tds-llm-analysis.s-anand.net/demo"
    answer: str = "anything you want"




llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
    api_key=KEY
)


class State(TypedDict):
    """Graph state containing conversation messages."""
    messages: Annotated[list, add_messages]




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

llm_with_tools = llm.bind_tools(tools)


def chatbot(state: State):
    """Agent node that processes messages and decides whether to use tools."""
    return {"messages": [llm_with_tools.invoke(state["messages"])]}


def route_tools(state: State):
    """Route to tools if the last message has tool calls, otherwise end."""
    if isinstance(state, list):
        ai_message = state[-1]
    elif messages := state.get("messages", []):
        ai_message = messages[-1]
    else:
        raise ValueError("No messages found in input")
    
    if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
        return "tools"
    return END


graph_builder = StateGraph(State)

graph_builder.add_node("chatbot", chatbot)
tool_node = ToolNode(tools=tools)
graph_builder.add_node("tools", tool_node)

graph_builder.add_edge(START, "chatbot")
graph_builder.add_conditional_edges("chatbot", route_tools, ["tools", END])
graph_builder.add_edge("tools", "chatbot")


graph = graph_builder.compile()


def run_agent(query: str, system_prompt: str|None = None,reason=None):
    messages = []

    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    if reason:
        query+=f"Previously provided answer is wrong and here is the reason {reason}"
    messages.append({"role": "user", "content": query})
    result = graph.invoke({"messages": messages})
    for message in result["messages"]:
        if hasattr(message, "content") and isinstance(message.content, str):
            try:
                import json
                parsed = json.loads(message.content)
                if "next_url" in parsed and "answer" in parsed:
                    return {
                        "submit_url": parsed["next_url"],
                        "answer": parsed["answer"]
                    }
            except:
                pass
    final_message = result["messages"][-1]
    return final_message.content


if __name__ == "__main__":
    import json
    
    while True:
        q = input("\nAsk (or 'exit' to quit): ")
        if q.lower() in ("exit", "quit"):
            break
        
        try:
            result = run_agent(
                f"Solve the quiz at this URL: {q}",
                system_prompt="""You are a quiz solving agent with proper tooling.
After solving the quiz, you MUST call the submit_answer tool with:
- next_url: the submission URL
- answer: the quiz answer
Do not try to post the result yourself. Always use the submit_answer tool to return your final result."""
            )
            if isinstance(result, dict):
                print(json.dumps(result, indent=2))
            else:
                print(result)
        except Exception as e:
            print(f"\nERROR: {e}")
            import traceback
            traceback.print_exc()






prompt="""
You are an intelligent quiz-solving agent equipped with specialized tools for web scraping, data analysis, and API interaction.

**Your Mission:**
Analyze the given URL, extract the quiz question and any required data, solve it accurately, and submit your answer using the submit_answer tool.

**Step-by-Step Approach:**
1. **Scrape the page** - Use scrape_js tool to extract the quiz question, instructions, and any embedded data
2. **Gather required data** - If the quiz references external APIs, PDFs, or CSV data:
   - Use fetch_api for API endpoints
   - Use extract_pdf for PDF documents
   - Use clean_data and analyze_data for CSV processing
3. **Solve the problem** - Apply logical reasoning, mathematical operations, or data analysis as needed
4. **Submit your answer** - MUST call submit_answer tool with:
   - next_url: The submission URL (usually ends with /submit)
   - answer: Your calculated/derived answer (as a string)

**Critical Rules:**
- Always extract the submission URL from the page or construct it properly
- Provide answers in the exact format requested (number, text, JSON, etc.)
- Double-check your calculations before submitting
- Use tools sequentially and purposefully - don't skip steps
- Never fabricate answers - if you can't solve it, explain why

**Tool Usage Guidelines:**
- scrape_js: For any webpage content extraction
- fetch_api: For JSON/API endpoints
- extract_pdf: For PDF text extraction
- clean_data: For CSV cleaning operations
- analyze_data: For mean/sum calculations on CSV data
- plot_data: For visualization (when needed)
- submit_answer: FINAL step - returns structured result

Remember: Your goal is accuracy and proper tool usage. Think through each step carefully.
"""


SECRET = os.getenv("secret")
t:int =int(os.getenv("retry",3))

def solve(url, email, reason=None):
    res = run_agent(url, system_prompt=prompt, reason=reason)
    print(res)
    
    # Handle case where res is a string (agent didn't use submit_answer tool)
    if isinstance(res, str):
        print(f"Warning: Agent returned string instead of dict: {res}")
        return None
    
    # Validate that res is a dict with required keys
    if not isinstance(res, dict) or "submit_url" not in res or "answer" not in res:
        print(f"Error: Invalid response format: {res}")
        return None
    
    submit_url = res["submit_url"]
    answer = res["answer"]
    
    for _ in range(t):
        try:
            print(SECRET)
            response = httpx.post(
                submit_url,
                json={
                    "email": email,
                    "secret": SECRET,
                    "url": url,
                    "answer": answer
                },
                timeout=10
            )
            print(response.json())
            return response
        except httpx.HTTPError as e:
            print(e)
    
    return None


def solve_quiz(quiz: Quiz):
    current_url = quiz.url
    email = quiz.email
    reason = None
    print(SECRET)
    print(quiz)
    
    while current_url:
        attempts = 0
        print(current_url)
        
        while attempts < t:
            res = solve(current_url, email, reason)
            
            # Check if solve returned None (error case)
            if res is None:
                print(f"Error: solve() returned None for URL: {current_url}")
                attempts += 1
                continue
            
            try:
                data = QuizResponse.model_validate(res.json())
            except Exception as e:
                print(f"Error validating response: {e}")
                attempts += 1
                continue
            
            if data.correct:
                reason = None  # Reset reason on success
                current_url = data.url
                break
            
            reason = data.reason
            attempts += 1
            
            if data.url:
                current_url = data.url
                break
        
        if attempts == t and not data.correct and not data.url:  # type: ignore
            print(f"Max attempts reached for URL: {current_url}")
            return
        
        if current_url is None:
            print("Quiz completed successfully!")
            return











@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    return JSONResponse(
        status_code=400,
        content={"detail": exc.errors()},
    )


@app.post("/solve",response_model=Quiz)
def solve_api(task:Quiz,background_tasks: BackgroundTasks):
    if task.secret == SECRET:
        print(task)
        background_tasks.add_task(solve_quiz,task)
        return Response(status_code=204)
    else:
        raise HTTPException(status_code=403, detail="Forbidden")

@app.post("/test",response_model=Test)
def test(test:Test):
    res = httpx.post(test.submit_url,
                     json=
                     {
                      "email": "your email",
                      "secret": "your secret",
                      "url": "https://tds-llm-analysis.s-anand.net/demo",
                      "answer": "anything you want"
                    }
                    )
    data = res.json()
    test_url=data["url"]
    q = Quiz(
        url=test_url,
        email="your email",
        secret="your secret")
    print(SECRET)
    print(data)
    solve_quiz(q)
    return Response(status_code=200)