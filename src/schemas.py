from pydantic import BaseModel

class Input(BaseModel):
  major : str
  tech: str
  