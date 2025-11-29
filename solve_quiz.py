from test_tool_output import run_agent
import httpx
import os
from dotenv import load_dotenv
from app.schema.schema import Quiz,QuizResponse
load_dotenv()

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


SECRET = os.getenv("secret","your secret")
t:int =int(os.getenv("retry",3))

def solve(url,email,reason=None):
    res = run_agent(url,system_prompt=prompt,reason=reason)
    print(res)
    submit_url = res["submit_url"]
    answer = res["answer"]
    for _ in range(t):
        try:
            print(SECRET)
            response = httpx.post(
                submit_url,
                json={
                    "email":email,
                    "secret":SECRET,
                    "url":url,
                    "answer":answer
                },
            timeout=10
            )
            print(response.json())
            return response
        except httpx.HTTPError as e:
            print(e)


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
            data = QuizResponse.model_validate(res.json()) #type:ignore
            if data.correct:
                reason = data.reason
                current_url = data.url
                break
            reason = data.reason
            attempts += 1
            if data.url:
                current_url = data.url
                break
        if attempts == t and not data.correct and not data.url:#type:ignore
            return
        if current_url is None:
            return
