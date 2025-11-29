from pydantic import BaseModel
from typing import Optional

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
