
import os
import json
import re
from datetime import date

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

app = FastAPI()

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:latest")


class InvoiceRequest(BaseModel):
    text: str = Field(min_length=1)


class InvoiceResponse(BaseModel):
    vendor: str
    amount: float
    currency: str
    date: str

    @field_validator("vendor")
    @classmethod
    def validate_vendor(cls, value):
        if not value.strip():
            raise ValueError("Vendor cannot be empty")
        return value.strip()

    @field_validator("currency")
    @classmethod
    def validate_currency(cls, value):
        value = value.strip().upper()
        if not re.fullmatch(r"[A-Z]{3}", value):
            raise ValueError("Invalid currency")
        return value

    @field_validator("date")
    @classmethod
    def validate_date(cls, value):
        value = str(value).strip()
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
            raise ValueError("Invalid date format")
        date.fromisoformat(value)
        return value


SYSTEM_PROMPT = """
You are a precise invoice information extraction engine.

Extract exactly these four fields from the invoice:
- vendor: the full vendor or supplier company name
- amount: the FINAL TOTAL AMOUNT DUE, as a number
- currency: the 3-letter uppercase currency code
- date: the PAYMENT DUE DATE in YYYY-MM-DD format

Important rules:
1. Do not confuse invoice date with payment due date.
2. Do not confuse subtotal, taxes, or item prices with total due.
3. Preserve the full vendor name including suffixes such as Ltd.
4. Currency may be written as USD, EUR, GBP, $, €, or £.
5. Convert $ to USD, € to EUR, and £ to GBP.
6. Convert written dates to YYYY-MM-DD.
7. Never invent missing values.
8. Return ONLY a valid JSON object.
9. No markdown, explanations, or extra fields.

Required example:
{
  "vendor": "Example Industries Ltd.",
  "amount": 1250.50,
  "currency": "USD",
  "date": "2026-12-15"
}
"""


@app.get("/")
def home():
    return {"status": "ok", "service": "invoice-extractor"}


@app.post("/extract", response_model=InvoiceResponse)
async def extract_invoice(request: InvoiceRequest):

    invoice_text = request.text.strip()

    if not invoice_text:
        raise HTTPException(
            status_code=422,
            detail="Invoice text cannot be empty"
        )

    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": invoice_text
            }
        ],
        "stream": False,
        "format": "json",
        "options": {
            "temperature": 0
        }
    }

    try:
        async with httpx.AsyncClient(timeout=90.0) as client:
            response = await client.post(
                OLLAMA_URL.rstrip("/") + "/api/chat",
                json=payload
            )

            response.raise_for_status()

        result = response.json()
        content = result["message"]["content"]
        extracted = json.loads(content)

        return InvoiceResponse.model_validate(extracted)

    except (httpx.HTTPError, KeyError, ValueError, TypeError):
        raise HTTPException(
            status_code=503,
            detail="Invoice extraction failed or Ollama unavailable"
        )
