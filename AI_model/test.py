import pandas as pd
from sentence_transformers import SentenceTransformer, util
import torch

# 샘플 데이터: 기업 이름, 분야, 기술 스택
data = {
    'company_name': ['CompanyA', 'CompanyB', 'CompanyC', 'CompanyD', 'CompanyE'],
    'sector': ['IT', 'Healthcare', 'Finance', 'IT', 'IT'],
    'tech_stack': ['Python, AI, ML', 'Biotech, Research', 'FinTech, Blockchain', 'Java, Cloud, ML', 'Python, Cloud, AI']
}

# 데이터프레임 생성
df = pd.DataFrame(data)

# 사용자 입력: 분야와 기술 스택
input_sector = 'IT'
input_tech_stack = 'Python, AI, Cloud'

# 입력 정보를 결합한 텍스트 생성
input_combined = input_sector + ', ' + input_tech_stack

# 기업의 분야와 기술 스택을 결합하여 텍스트 필드 생성
df['combined_features'] = df['sector'] + ', ' + df['tech_stack']

# Sentence-BERT 모델 로드
model = SentenceTransformer('all-MiniLM-L6-v2')  # 경량화된 SBERT 모델

# 기업 데이터 임베딩
company_embeddings = model.encode(df['combined_features'].tolist(), convert_to_tensor=True)

# 사용자 입력 임베딩
input_embedding = model.encode(input_combined, convert_to_tensor=True)

# 코사인 유사도를 사용하여 입력 데이터와 기업 간 유사도 계산
cosine_scores = util.cos_sim(input_embedding, company_embeddings)

# 유사도를 기준으로 정렬하여 상위 기업 추천
top_n = 3
top_results = torch.topk(cosine_scores, k=top_n)

# 추천 기업 출력 (indices를 리스트로 변환하여 정수 인덱스로 사용)
recommended_companies = [df.iloc[i]['company_name'] for i in top_results.indices[0].tolist()]
recommended_companies.sort()
print(f"추천 기업 (기준: 분야={input_sector}, 기술 스택={input_tech_stack}): {recommended_companies}")
