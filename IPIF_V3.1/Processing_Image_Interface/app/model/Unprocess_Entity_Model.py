from pydantic import BaseModel, ValidationError, validator, Field, root_validator, model_validator
from typing import List, Dict, Any

class ValidationErrorDetail(BaseModel):
    loc: List[str]
    msg: str
    type: str
    input: Dict[str, Any]
    ctx: Dict[str, Any] = None
    url: str

class Unprocess_Entity_Model(BaseModel):
    detail: List[ValidationErrorDetail]