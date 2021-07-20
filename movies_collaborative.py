
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd

from nltk.tokenize import word_tokenize
import nltk
#nltk.download('punkt')

# TF-IDF 로 표현하기 item-matrix 부터?
# 비슷한 영화를 추천해주기
# 유사도 이용 추천시스템 구현
# TF-IDF 와 코사인 유사도만으로 영화 줄거리 기반 영화 추천 시스템

if __name__ == '__main__':


    credits = pd.read_csv('./movies/credits.csv')
    keywords = pd.read_csv('./movies/keywords.csv')
    links = pd.read_csv('./movies/links.csv')


    movies_metadata = pd.read_csv('./movies/movies_metadata.csv')

    # 유사도를 구해야된다 overview를 시퀀스로 변경
    # over_view = list(movies_metadata['overview'])
    print(movies_metadata.dtypes)
    movies_metadata.astype({'overview': 'str'})
    print(movies_metadata.dtypes)


    movies_metadata['tokenized_sents'] = movies_metadata.apply(lambda row: word_tokenize(row['overview']), axis=1)

#['tokenized_overview'] = movies_metadata['overview'].apply(lambda x : word_tokenize(x))
    print('EDA')
    # fit_transform은 사전 만들고 유사도 까지 구하는 것 .. 근데 그렇게 해도 상관은 없네 사실 담긴 의미를 해석하면
    # cnt_vect = CountVectorizer(min_df=0, ngram_range=(1,2))
    # text_vect = cnt_vect.fit_transform(movies_metadata['over_view'])

    # TF-IDF 만들기
    # list_= list(map(str.split(' '),over_view))

    #
    # print(list(map(word_tokenize,over_view)))



    # 모든 단어 토큰화

    #


    # 파이썬 구현
    # 불용어 제외하기
    stopwords = ['by', 'in' , 'A']



    over_view = list(movies_metadata['overview'].str.split())


#    over_view_word = [word for i in range(len(over_view)) for word in over_view[i]]




#    over_view_word = list(set(word for word in movies_metadata['overview'].split()))


    over_view_word = over_view_word.sort()






    print(over_view_word)

    #

    # overview 를 TF-IDF 로 표현하기
    # DTM 표현하기
    # 문서별 overview 표현









    print(movies_metadata)

