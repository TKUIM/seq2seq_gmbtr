#輸入新句
import jieba

jieba.set_dictionary('data/dict.txt')

newdata_path = 'test/SCtoCC.txt'
# test/ CCtoSC.txt SCtoCC.txt
# newdata.txt newdata-cs.txt
f = open(newdata_path, 'a', encoding='utf-8')
'''
r    讀取(檔案需存在)
w    新建檔案寫入(檔案可不存在，若存在則清空)
a    資料附加到舊檔案後面(游標指在EOF)
r+   讀取舊資料並寫入(檔案需存在且游標指在開頭)
w+   清空檔案內容，新寫入的東西可在讀出(檔案可不存在，會自行新增)
a+   資料附加到舊檔案後面(游標指在EOF)，可讀取資料
b    二進位模式
'''
#天將降大任於斯人也
#三人行必有我師焉
#夫天地者萬物之逆旅
#山不在高有仙則名
#水不在深有龍則靈

# 暫時取消輸入新句
#new_sentence = input()
#f.write(new_sentence + '\t' + ' \n')
#f.flush()
f.close()

'''
print('key "break" to exit')
new_sentence = ''
while True:
    new_sentence = input()
    if input() == 'break':
        f.flush()
        break
    #repeat_check = True
    f.write(new_sentence + '\t' + ' \n')
    f.flush()
f.close()
'''

#斷詞
output = open('test/newdata_seg.txt', 'w', encoding='utf-8')
# newdata_seg.txt
# 讀寫檔案，確定檔案一定會關閉，使用 with as 語句來簡化
with open(newdata_path, 'r', encoding='utf-8') as content :
    for texts_num, line in enumerate(content):
        line = line.strip('\n')
        words = jieba.cut(line, cut_all=False)
        for word in words:
            output.write(word + ' ')
        output.write('\n')
output.close()

#line切字元
#output = open('test/cht_lv_seg.txt', 'w', encoding='utf-8')
#test/cht_lv_seg.txt

#test/CCtoSC_ChLv.txt
with open('test/chtlv/ChtLv_SCtoCC_S20P_03.txt', 'r+', encoding='utf-8') as fr:
    #test/cht_lv_seg.txt
    with open('test/ChtLv_S2C_SP_03.txt','w', encoding='utf-8') as fw:
        fw.write(''.join([f+' ' for fh in fr for f in fh]))

#清理
import re
f_input = open('test/newdata_seg.txt', 'r+', encoding='utf-8')
f_output = open('test/newdata_clean.txt', 'w', encoding='utf-8')
line = f_input.readline()
# 用 while 逐行讀取檔案內容，直至檔案結尾
while line:
    #line = re.sub('^ ', '', line)
    line = re.sub(' +', ' ', line)
    f_output.write(line)
    print(line)
    line = f_input.readline()
f_input.close()
f_output.close()

#檢視
import re
f = open('test/newdata_clean.txt', 'r+', encoding='utf-8')
line = f.readline()
# 用 while 逐行讀取檔案內容，直至檔案結尾
while line:
    print(line)
    line = f.readline()
f.close()
