from fastapi import FastAPI,Request,HTTPException,Response,BackgroundTasks
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
import os
from dotenv import load_dotenv
from app.schema.schema import Quiz,Test
from solve_quiz import solve_quiz
import httpx

load_dotenv()

app = FastAPI()
SECRET = os.getenv("secret")



app = FastAPI()

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    return JSONResponse(
        status_code=400,
        content={"detail": exc.errors()},
    )


@app.post("/solve",response_model=Quiz)
def solve(task:Quiz,background_tasks: BackgroundTasks):
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