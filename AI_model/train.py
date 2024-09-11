import pandas as pd
from sentence_transformers import SentenceTransformer, util
import torch
from fastapi import  APIRouter
from src.schemas import Input  # Input 클래스가 src.schemas에 정의되어 있다고 가정


