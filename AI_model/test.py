import pandas as pd
from sentence_transformers import SentenceTransformer, util
import torch
from fastapi import  APIRouter
from src.schemas import Input  # Input 클래스가 src.schemas에 정의되어 있다고 가정

router = APIRouter()

# 테스트 데이터 임의 값
data = {
    'company_name': ['CompanyA', 'CompanyB', 'CompanyC', 'CompanyD', 'CompanyE'],
    'sector': ['IT', 'Healthcare', 'Finance', 'IT', 'IT'],
    'tech_stack': ['Python, AI, ML', 'Biotech, Research', 'FinTech, Blockchain', 'Java, Cloud, ML', 'Python, Cloud, AI']
}

# 데이터프레임 생성
df = pd.DataFrame(data)
df['combined_features'] = df['sector'] + ', ' + df['tech_stack']

# Sentence-BERT 모델 불러오기 
model = SentenceTransformer('all-MiniLM-L6-v2')

# 기업 데이터 임베딩(AI가 알아먹을 수 있는 디지털로 바꾸는 것)
company_embeddings = model.encode(df['combined_features'].tolist(), convert_to_tensor=True)

@router.post('/recommend')
async def recommend_company(input: Input):
    # 사용자 입력 결합
    input_combined = input.major + ', ' + input.tech

    # 사용자 입력 임베딩
    input_embedding = model.encode(input_combined, convert_to_tensor=True)

    # 코사인 유사도를 사용하여 입력 데이터와 기업 간 유사도 계산
    cosine_scores = util.cos_sim(input_embedding, company_embeddings)

    # 유사도를 기준으로 정렬하여 상위 기업 추천
    top_n = 3
    top_results = torch.topk(cosine_scores, k=top_n)

    # 추천 기업 추출
    recommended_companies = [df.iloc[i]['company_name'] for i in top_results.indices[0].tolist()]
    recommended_companies.sort() # 정렬

    # 추천 결과 반환
    return {"recommended_companies": recommended_companies}

