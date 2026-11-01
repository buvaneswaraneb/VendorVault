from pathlib import Path
from dotenv import load_dotenv
from google import genai
import os

from app.schemas import QueryRequest
from app.services.useage import CRUD
load_dotenv()

CURRENT_FILE = Path(__file__).resolve().parent # fetches the parent repo



def genrate_response(SETTINGS,QR:QueryRequest):
    client = genai.Client(api_key=SETTINGS.gemini_api_key)
    interaction = client.interactions.create(
        model = SETTINGS.primary_model,
        input= QR.query
    )

    # interactions useage is stored in the db 

    return 



    
