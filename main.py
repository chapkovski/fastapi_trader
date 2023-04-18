# Import the FastAPI library
from fastapi import FastAPI

# Create an instance of the FastAPI class
app = FastAPI()

# Define a route for the root URL ("/") using the get() decorator
@app.get("/")
def read_root():
    return {"message": "Hello, JOPA!"}

# Define another route with a path parameter
@app.get("/greet/{name}")
def greet(name: str):
    return {"message": f"pizdtes, {name}!"}
