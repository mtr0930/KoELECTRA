from sklearn.feature_extraction.text import CountVectorizer
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import pandas as pd
import numpy as np
import re

scope = [
    'https://spreadsheets.google.com/feeds',
    'https://www.googleapis.com/auth/drive',
]
json_file_name = 'C:/Users/mtr09/PycharmProjects/KoELECTRA/finetune/data/korquad/bustling-tuner-324606-a34f304d6112.json'
credentials = ServiceAccountCredentials.from_json_keyfile_name(json_file_name, scope)
gc = gspread.authorize(credentials)
spreadsheet_url = 'https://docs.google.com/spreadsheets/d/1icMFhFRT8L2vBAHvgleP2y6qFtamBTXbiJSTb5rCoWA/edit#gid=0'
# 스프레스시트 문서 가져오기
doc = gc.open_by_url(spreadsheet_url)
# 시트 선택하기
parts = ["STRATEGY", "AUTHORITY", "TYPE", "DEFINITION", "CENTER", "PROFILE"]

# 126개 paragraph만 저장할 배열
paragraphs = []

# 126개 paragraph 별 title, part 정보 저장하기 위한 배열
paragraph_info = []

# 질문, 정답 paragraph, title, part 저장하기 위한 배열
sheet_questions_answers = []

# 전체 질문 수 count
total_count = 0
for part in parts:
    worksheet = doc.worksheet(part)
    # 1열 title
    column1_data = worksheet.col_values(1)
    column1_data = column1_data[1:]
    # 2열은 아무것도 없고
    # 3열 paragraph
    column3_data = worksheet.col_values(3)
    column3_data = column3_data[1:]
    # 4열 question
    column4_data = worksheet.col_values(4)
    column4_data = column4_data[1:]
    # 5열 질문에 해당하는 paragraph
    column5_data = worksheet.col_values(5)
    column5_data = column5_data[1:]

    for i in range(len(column4_data)):
        total_count += 1
        sheet_questions_answers.append([column4_data[i], column5_data[i], column1_data[i], part])
    # index
    s = 0
    for data in column3_data:
        if data != '':
            paragraphs.append(data)
            paragraph_info.append([data, column1_data[s], part])
        s += 1

print("total question 수 : ", total_count)
# model, tokenizer load
tokenizer = AutoTokenizer.from_pretrained("mtr0930/koelectra-base-v3_epoch-100")
model = AutoModelForQuestionAnswering.from_pretrained("mtr0930/koelectra-base-v3_epoch-100", return_dict=False)


def dummy(doc):
    return doc


def custom_tokenizer(text):
    my_inputs = tokenizer(text, add_special_tokens=False, return_tensors="pt")
    my_input_ids = my_inputs["input_ids"].tolist()[0]
    my_text_tokens = tokenizer.convert_ids_to_tokens(my_input_ids)
    # index
    c = 0
    for my_token in my_text_tokens:

        if '#' in my_token:
            string1 = my_token
            string1 = re.sub("#", "", string1)
            my_text_tokens[c] = string1

        c += 1

    for tmp_token in my_text_tokens:
        if tmp_token == '.' or tmp_token == ',' or tmp_token == "'" or tmp_token == '?':
            my_text_tokens.remove(tmp_token)

    return my_text_tokens

# tf-idf matrix 변환하기 위해 입력으로 들어갈 토큰화된 text 배열
tokenized_texts = []

for text in paragraphs:
    inputs = tokenizer(text, add_special_tokens=False, return_tensors="pt")
    input_ids = inputs["input_ids"].tolist()[0]
    text_tokens = tokenizer.convert_ids_to_tokens(input_ids)
    d = 0
    for token in text_tokens:
        # '#'문자 제거
        if '#' in token:
            string2 = token
            string2 = re.sub("#", "", string2)
            text_tokens[d] = string2
        d += 1
    # context 안에 특수 문자 제거
    for t_token in text_tokens:
        if t_token == '.' or t_token == ',' or t_token == "'" or t_token == "?":
            text_tokens.remove(t_token)

    tokenized_texts.append(text_tokens)

# ============================================ 여기서 부터 tf-idf matrix 적용 =====================================================


vect = CountVectorizer(max_features=10000, tokenizer=dummy, preprocessor=dummy)

document_term_matrix = vect.fit_transform(tokenized_texts)  # 문서-단어 행렬
feature_names = vect.get_feature_names()
print(document_term_matrix.shape)

tf = pd.DataFrame(document_term_matrix.toarray(), columns=vect.get_feature_names())
column_names = tf.columns
# TF (Term Frequency)
D = len(tf)
df = tf.astype(bool).sum(axis=0)
idf = np.log((D + 1) / (df + 1)) + 1  # IDF (Inverse Document Frequency)

# TF-IDF (Term Frequency-Inverse Document Frequency)
tfidf = tf * idf
tfidf = tfidf / np.linalg.norm(tfidf, axis=1, keepdims=True)



# ============================================ 여기까지 tf-idf matrix 적용 =====================================================

result = []
# 정답 수
count = 0
just_once = False
# 잘못된 답변 기록하기 위한 파일

f_ans = open("correct_answer.txt", 'w', encoding='utf-8')
f_st = open("wrong_same_title.txt", 'w', encoding='utf-8')
f_sp = open("wrong_same_part.txt", 'w', encoding='utf-8')
f_dp = open("wrong_diff_part.txt", 'w', encoding='utf-8')
# 오답 케이스 count하기 위한 변수
same_title = 0
same_part = 0
diff_part = 0

# ============================================ 질문 tf-idf vector로 변환 적용 =====================================================
for qas in sheet_questions_answers:
    question = qas[0]
    answer = qas[1]
    title = qas[2]
    part_name = qas[3]
    question = custom_tokenizer(question)
    original_question = question
    # 질문 vector
    one_coded_list = []
    # 토큰화한 질문에서 feature토큰이 있다면 1, 없다면 0
    # 2차원 vector size = (feature 토큰 수, 1) 
    for feature in feature_names:
        token_found = False
        for token in question:
            if token == feature:
                token_found = True
                one_coded_list.append(1)
                break
        if token_found == False:
            one_coded_list.append(0)

    question_array = np.array(one_coded_list)
    tfidf_array = tfidf.to_numpy()

    dot_result = np.dot(tfidf_array, question_array)
    max_index = dot_result.argmax(axis=0)
    # score 가 가장 높은 paragraph
    max_predic = paragraphs[max_index]
    # 정답일 경우
    if max_predic == answer:
        f_ans.write(f"Original question : {original_question}\n\n")
        f_ans.write(f"Original context : {answer}\n\n")
        f_ans.write("========================\n")
        count += 1

    # 오답일 경우
    else:
        # 오답중 같은 title 일 경우
        if title == paragraph_info[max_index][1]:
            f = f_st
            same_title += 1
        # 오답중 다른 title 일 경우
        else:
            # 같은 part 인 경우
            if part_name == paragraph_info[max_index][2]:
                f = f_sp
                same_part += 1
            # title 도 다르고 part 도 다른 경우
            else:
                f = f_dp
                diff_part += 1
        f.write(f"Original question : {original_question}\n")
        f.write(f"Original context : {answer}\n")
        f.write(f"Original part : {part_name}\n")
        f.write(f"Original title : {title}\n\n")


        f.write(f"Predicted context : {max_predic}\n")
        f.write(f"Predicted part : {paragraph_info[max_index][2]}\n")
        f.write(f"Predicted title : {paragraph_info[max_index][1]}\n")
        f.write("========================\n\n")

# 정확도
accuracy = (count / total_count) * 100
print(f"same title : {same_title}, same part : {same_part}, diff part : {diff_part}")
print("정확도 : %.2f" % accuracy, "%")


