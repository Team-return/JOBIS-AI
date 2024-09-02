from fastapi import FastAPI
import uvicorn
from AI_model.test import router

app = FastAPI(title="Jobis AI")

@app.get("/")
def hi():
  return{"message" : "안녕하세요 자비스 AI 입니다"}


app.include_router(router)


if __name__ == "__main__":
  uvicorn.run(app="main:app", host= "0.0.0.0" , port=8000)

