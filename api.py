from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def inicio():
    return {"mensaje": "API AgroSignal funcionando"}

@app.get("/dolar-blue")
def dolar():
    return {
        "compra": 1300,
        "venta": 1320
    }

@app.get("/senal")
def senal():
    return {
        "cultivo": "Soja",
        "senal": "VENDER",
        "confianza": 0.91
    }
