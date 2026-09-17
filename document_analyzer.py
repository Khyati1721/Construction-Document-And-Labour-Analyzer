from pathlib import Path
import json
import mimetypes
import os

from google import genai
from google.genai import types


# GEMINI PROMPT
PROMPT = """
You are a construction document analyzer.

Analyze the uploaded construction document carefully.

The document may be:
- Material challan
- Supplier invoice
- Purchase bill
- Labour bill
- Delivery receipt
- Other construction-related document

Extract the information into the requested JSON structure.

Important rules:

1. Do not invent information.
2. If a text field is not available, return an empty string.
3. If a numeric field is not available, return 0.
4. Extract all visible line items.
5. Preserve quantities, units, rates and amounts exactly as shown
   on the document when they are clearly visible.
6. Do NOT calculate or validate bill totals.
7. Do NOT calculate quantity × rate yourself.
8. For the item "amount", extract the amount printed on the document.
9. Extract subtotal, GST amount and grand total only when they are
   explicitly visible on the document.
10. Read the document carefully, including tables.
"""


# GEMINI RESPONSE SCHEMA
SCHEMA = {
    "type": "object",
    "properties": {

        "document_type": {
            "type": "string",
            "description": "Type of construction document"
        },

        "supplier_or_contractor": {
            "type": "string",
            "description": "Supplier or contractor name"
        },

        "document_number": {
            "type": "string",
            "description": "Invoice, challan or document number"
        },

        "date": {
            "type": "string",
            "description": "Document date"
        },

        "vehicle_number": {
            "type": "string",
            "description": "Vehicle registration number"
        },

        "gstin": {
            "type": "string",
            "description": "GSTIN"
        },

        "bill_to": {
            "type": "string",
            "description": "Bill-to customer"
        },

        "transporter": {
            "type": "string",
            "description": "Transporter name"
        },

        "delivery_date": {
            "type": "string",
            "description": "Delivery date"
        },

        # ITEMS
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {

                    "description": {
                        "type": "string",
                        "description": "Material or item description"
                    },

                    "quantity": {
                        "type": "number",
                        "description": "Quantity shown on document"
                    },

                    "unit": {
                        "type": "string",
                        "description": "Unit shown on document"
                    },

                    "rate": {
                        "type": "number",
                        "description": "Rate shown on document"
                    },

                    "amount": {
                        "type": "number",
                        "description": "Amount printed on document"
                    }

                },
                "required": [
                    "description",
                    "quantity",
                    "unit",
                    "rate",
                    "amount"
                ]
            }
        },

        # DOCUMENT TOTALS

        "subtotal": {
            "type": "number",
            "description": "Subtotal printed on document"
        },

        "gst_amount": {
            "type": "number",
            "description": "GST amount printed on document"
        },

        "grand_total": {
            "type": "number",
            "description": "Grand total printed on document"
        },

        "currency": {
            "type": "string",
            "description": "Currency shown on document"
        }
    },

    "required": [
        "document_type",
        "supplier_or_contractor",
        "document_number",
        "date",
        "vehicle_number",
        "gstin",
        "bill_to",
        "transporter",
        "delivery_date",
        "items",
        "subtotal",
        "gst_amount",
        "grand_total",
        "currency"
    ]
}


# API KEY
def get_api_key():

    try:
        import streamlit as st

        if "GEMINI_API_KEY" in st.secrets:
            return st.secrets["GEMINI_API_KEY"]

    except Exception:
        pass

    return os.getenv(
        "GEMINI_API_KEY"
    )


# GEMINI MODEL
def get_model():

    try:
        import streamlit as st

        if "GEMINI_MODEL" in st.secrets:
            return st.secrets["GEMINI_MODEL"]

    except Exception:
        pass

    return os.getenv(
        "GEMINI_MODEL",
        "gemini-2.5-flash"
    )


# DOCUMENT ANALYZER
def analyze_document(document_path: str):
    """
    Analyze a construction document using Google Gemini.

    Supports:
    - PDF
    - JPG
    - JPEG
    - PNG
    - WEBP

    This function only extracts information.
    It does NOT perform bill calculations or validation.
    """

    # API KEY
    api_key = get_api_key()

    if not api_key:

        raise RuntimeError(
            "GEMINI_API_KEY is not configured. "
            "Add it to .streamlit/secrets.toml."
        )

    # DOCUMENT PATH
    document_path = Path(
        document_path
    )

    if not document_path.exists():

        raise FileNotFoundError(
            f"Document not found: {document_path}"
        )

    # GEMINI CLIENT
    model = get_model()

    client = genai.Client(
        api_key=api_key
    )

    suffix = document_path.suffix.lower()

    # PDF
    if suffix == ".pdf":

        uploaded_file = client.files.upload(
            file=str(document_path)
        )

        response = client.models.generate_content(
            model=model,

            contents=[
                uploaded_file,
                PROMPT
            ],

            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=SCHEMA,
            ),
        )

    # IMAGE
    else:

        mime_type, _ = mimetypes.guess_type(
            str(document_path)
        )

        if not mime_type:

            mime_type = "image/jpeg"

        image_bytes = document_path.read_bytes()

        image_part = types.Part.from_bytes(
            data=image_bytes,
            mime_type=mime_type,
        )

        response = client.models.generate_content(
            model=model,

            contents=[
                PROMPT,
                image_part
            ],

            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=SCHEMA,
            ),
        )

    # CHECK GEMINI RESPONSE
    if not response.text:

        raise RuntimeError(
            "Gemini returned an empty response."
        )

    # PARSE JSON
    try:

        result = json.loads(
            response.text
        )

    except json.JSONDecodeError as e:

        raise RuntimeError(
            f"Gemini returned invalid JSON: {e}\n\n"
            f"Response:\n{response.text}"
        )

    # RETURN EXTRACTED DOCUMENT DATA
    return {
        "document": {
            "document_type": result.get(
                "document_type",
                ""
            ),

            "supplier_or_contractor": result.get(
                "supplier_or_contractor",
                ""
            ),

            "document_number": result.get(
                "document_number",
                ""
            ),

            "date": result.get(
                "date",
                ""
            ),

            "vehicle_number": result.get(
                "vehicle_number",
                ""
            ),

            "gstin": result.get(
                "gstin",
                ""
            ),

            "bill_to": result.get(
                "bill_to",
                ""
            ),

            "transporter": result.get(
                "transporter",
                ""
            ),

            "delivery_date": result.get(
                "delivery_date",
                ""
            ),
        },

        "items": result.get(
            "items",
            []
        ),

        "totals": {
            "subtotal": result.get(
                "subtotal",
                0
            ),

            "gst_amount": result.get(
                "gst_amount",
                0
            ),

            "grand_total": result.get(
                "grand_total",
                0
            ),

            "currency": result.get(
                "currency",
                ""
            ),
        },

        "ai_extracted": result,
    }