import jieba
import logging

def main():

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    
    #詞典
    jieba.set_dictionary('data/dict.txt') #官方詞典：非簡體原版，經繁體化
    jieba.load_userdict("data/userdict.txt") #自訂詞典
    
    #停用字表
    stopword_set = set()
    with open('data/stopwords_Extend.txt','r', encoding='utf-8') as stopwords:
        for stopword in stopwords:
            stopword_set.add(stopword.strip('\n'))

    #斷詞
    output = open('wiki_sc_seg.txt', 'w', encoding='utf-8')
    with open('data/wiki_zh_tw.txt', 'r', encoding='utf-8') as content :
        for texts_num, line in enumerate(content):
            line = line.strip('\n')
            words = jieba.cut(line, cut_all=False)
            for word in words:
                if word not in stopword_set:
                    output.write(word + ' ')
            output.write('\n')

            if (texts_num + 1) % 10000 == 0:
                logging.info("已處理 %d 行斷詞" % (texts_num + 1))
    logging.info("已完成全部斷詞")
    output.close()

if __name__ == '__main__':
    main()
