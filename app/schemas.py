"""Request and response models. Empty for now; you add these from Session 4."""

from pydantic import BaseModel

class QueryRequest(BaseModel):
    query:str

class QueryResponse(BaseModel):
    answer:str
    model_used:str

class UseageResponse(BaseModel):
    total_tokens:int
    prompt_tokens:int
    completion_tokens:int
    estimated_cost_usd:int 
    estimated_cost_inr:int
    