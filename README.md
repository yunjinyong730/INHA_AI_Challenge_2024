# INHA_AI_Challenge_2024
<img width="1188" alt="스크린샷 2024-08-13 오후 4 27 38" src="https://github.com/user-attachments/assets/410fc7ec-2d7f-4981-801a-27486114de99">
 한국 경제 기사 분석 및 질의응답 | 알고리즘 | 언어 | 생성형 AI | LLM | QA | F1 Score


**[주관 / 운영]**

- 주관: 인공지능융합연구센터, BK 산업융합형 차세대 인공지능 혁신인재 교육연구단
- 후원: 포티투마루(42MARU)
- 운영: 데이콘

## Dataset Info

```jsx
train.csv [파일]
id : 샘플 고유 ID
context : 금융 및 경제 뉴스 기사 관련 정보
question : context 정보 기반 질문
answer : 질문에 대한 정답 (Target)
33716개 샘플

test.csv [파일]
id : 샘플 고유 ID
context : 금융 및 경제 뉴스 기사 관련 정보
question : context 정보 기반 질문
1507개 샘플

sample_submission.csv [파일] - 제출 양식
id : 샘플 고유 ID
question : context 정보 기반 질문
```

## 시도 모델

***beomi/llama-2-ko-7b***

```jsx
# llama-2-ko-7b 모델 로드
base_model = "beomi/llama-2-ko-7b"

model = AutoModelForCausalLM.from_pretrained(
    base_model,
    quantization_config=quant_config,
    device_map="auto"
    #device_map={"": 0}	# 0번째 gpu 에 할당
)
# 모델의 캐시 기능을 비활성화 한다. 캐시는 이전 계산 결과를 저장하기 때문에 추론 속도를 높이는 역할을 한다. 그러나 메모리 사용량을 증가시킬 수 있기 때문에, 메모리부족 문제가 발생하지 않도록 하기 위해 비활성화 해주는 것이 좋다.
model.config.use_cache = False
# 모델의 텐서 병렬화(Tensor Parallelism) 설정을 1로 지정한다. 설정값 1은 단일 GPU에서 실행되도록 설정 해주는 의미이다.
model.config.pretraining_tp = 1
```

***google/gemma-2b-it***

```jsx
BASE_MODEL = "google/gemma-2b-it"

model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, device_map="auto", quantization_config=bnb_config)
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, add_special_tokens=True)
tokenizer.padding_side = 'right'
```

## Method

- **fine tuning**
    - llama
      <img width="1196" alt="스크린샷 2024-08-13 오후 4 22 11" src="https://github.com/user-attachments/assets/0293c93e-5471-4b2c-ba2a-b9e04e4ffba2">



        
    - gemma
      <img width="1105" alt="스크린샷 2024-08-13 오후 4 21 02" src="https://github.com/user-attachments/assets/be4bd9dd-dc52-4984-b23f-914b2021257b">



        

- Data Augmentation

```jsx
#데이터 증강 함수 정의
def augment_data(dataframe):
    augmented_data = []

    # 동의어 대체를 위한 Augmenter 생성
    synonym_augmenter = naw.SynonymAug(aug_src='wordnet', aug_min=1)

    # 랜덤 교체를 위한 Augmenter 생성
    random_swap_augmenter = naw.RandomWordAug(action="swap", aug_p=0.1)

    for index, row in dataframe.iterrows():
        context = row['context']
        question = row['question']
        answer = row['answer']

        # 질문 문장을 동의어 대체하여 증강
        augmented_question_synonym = synonym_augmenter.augment(question)
        augmented_data.append({'context': context, 'question': augmented_question_synonym, 'answer': answer})

        # 질문 문장을 랜덤 교체하여 증강
        augmented_question_swap = random_swap_augmenter.augment(question)
        augmented_data.append({'context': context, 'question': augmented_question_swap, 'answer': answer})

        # 추가적으로 다른 증강 기법 적용 가능

    augmented_dataframe = pd.DataFrame(augmented_data)
    return augmented_dataframe
```

- Data Preprocessing(Cleaning)

```jsx
file_path = '/content/train.csv'
train_data = pd.read_csv(file_path)

# 데이터 전처리: 중복 제거, 결측값 처리, 텍스트 정규화
train_data.drop_duplicates(inplace=True)
train_data.dropna(subset=['context', 'question', 'answer'], inplace=True)

def clean_text(text):
    text = text.lower()  # 소문자 변환
    text = unicodedata.normalize("NFKD", text) # 유니코드 정규화 (발음 구별 기호 등 제거)
    text = contractions.fix(text) # 축약어 복원
    text = re.sub(r'\s+', ' ', text)  # 여러 공백을 하나의 공백으로 변환
    text = re.sub(f"[{string.punctuation}]", '', text)  # 구두점 제거
    return text

train_data['context'] = train_data['context'].apply(clean_text)
train_data['question'] = train_data['question'].apply(clean_text)
train_data['answer'] = train_data['answer'].apply(clean_text)

train_data = train_data.sample(frac=1).reset_index(drop=True)

#여기도 개선 가능!
val_data=train_data[:10]
train_data = train_data[2000:5001]

val_label_df = val_data[['question', 'answer']]

train_data.head(5)
```

## Processing


https://github.com/user-attachments/assets/022aa4c3-1a68-4e04-b6ca-0f2917aacd43




## 결과:

fine-tuning시에 파라미터를 지속적으로 조정하며 f1-score 50%까지 개선
<img width="1237" alt="스크린샷 2024-08-14 오후 5 52 16" src="https://github.com/user-attachments/assets/8c06936d-3fe2-49db-9e31-66f66ca8c0b5">
