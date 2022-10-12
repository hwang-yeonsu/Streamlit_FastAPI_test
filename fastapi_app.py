from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import uvicorn
import time

app = FastAPI()



@app.get("/")
def root():
    d = {
        "userId": 1,
        "id": 1,
        "title": "delectus aut autem",
        "completed": False,
    }
    return d


@app.get("/psi")
def psi():
    def generator():
        for i in range(3):
            yield "some fake lines\n"
            time.sleep(0.5)
        yield 'psi done.'

    return StreamingResponse(generator())


# if __name__=='__main__':
#     uvicorn.run('fastapi_app:app', reload=True)