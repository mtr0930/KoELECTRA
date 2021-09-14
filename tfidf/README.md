# TF-IDF 적용 모듈

## How to use
* Model, Tokenizer, Config 가져오기
```python
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

#KoELECTRA, mBERT 공통
config = AutoConfig.from_pretrained(model_name_or_path)
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

#KoELECTRA
model = AutoModelForQuestionAnswering.from_pretrained(model_name_or_path, return_dict=False)

#mBERT 모델
model = AutoModelForQuestionAnswering.from_pretrained(model_name_or_path, config=config)
```

* tfidf.py 사용법
```python
python tfidf.py --model_name_or_path mtr0930/i-manual_tokenizer_updated --data_path ./paragraph.json
```
## 기존 모듈과의 차이점
* data_path로 i-manual data에서 paragraph만 추출한 json파일을 넣어주어야함 -> paragraph.json
* 기존의 모델에서 특수 토큰 추가한 tokenizer로 update -> mtr0930/i-manual_tokenizer_updated

## Updated Tokenizer
https://huggingface.co/mtr0930/i-manual_tokenizer_updated
