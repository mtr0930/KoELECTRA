# TF-IDF 적용 모듈
-------------------
## How to use
* Model, Tokenizer, Config 가져오기
```
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
```
python tfidf.py --model_name_or_path mtr0930/i-manual_tokenizer_updated --data_path ./paragraph.json
```
