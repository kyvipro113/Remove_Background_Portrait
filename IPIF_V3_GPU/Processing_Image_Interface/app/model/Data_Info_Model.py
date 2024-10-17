from pydantic import BaseModel, model_validator

class Data_Info_Model(BaseModel):
    data_in: str
    data_out: str

    @model_validator(mode="after")
    def validate_fields(cls, v):
        if(v.data_in == ""):
            raise ValueError("Data in not found")
        if(v.data_out == ""):
            raise ValueError("Data out not found")
        return v
    
class Data_Info_Rtn_Model(Data_Info_Model):
    status: str = ""