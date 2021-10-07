# 형태소 품사중 실질형태소에 해당하는 부분만 추출

from konlpy.tag import Mecab

def mecab_normalize(text):
    ###################################################################
    # 실질 형태소 품사
    meaning_tags = ["NNG", "NNP", "NNB", "NR", "NP", "VV", "VA", "VX", "VCP", "VCN", "MM", "MAG", "MAJ", "IC"]
    
    # 실질 형태소 + 어근, 접미사, 접두사
    meaning_tags = ["NNG", "NNP", "NNB", "NR", "NP", "VV", "VA", "VX", "VCP", "VCN", "MM", "MAG", "MAJ", "IC", "XR", "XPN", "XSN", "XSV", "XSA"]
    
    ########### 둘중 원하는 방식으로 사용하면 됩니다. ##################
    
    # 딕셔너리 파일 위치는 사용자 별로 상이
    m = Mecab('C:\\mecab\\mecab-ko-dic')
    
    out = m.pos(text)
    
    # 실질 형태소만 담는 배열
    meaning_words = []
    
    for word in out:
        # word의 형태는 ("단어", "품사") 이므로 word[1]은 품사를 나타내고 word[0]는 단어를 나타냄.
        if word[1] in meaning_tags:
            meaning_words.append(word[0])

    return meaning_words
