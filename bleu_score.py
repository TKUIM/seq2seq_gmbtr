# sentence_bleu() 根據一句或多句參考句來評估候選句
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import modified_precision

#參照句
data_path = 'test/SCtoCC_clean.txt'
# CCtoSC_ChLv SCtoCC_ChLv .txt
# CCtoSC_clean.txt SCtoCC_clean.txt
# cc2sc_seg_clean.txt newdata_clean.txt

#候選句
trained_path = 'test/SC2CC_CV10_TR_03.txt'
# ChtLv_CCtoSC_CV10_01 ChtLv_C2S_CV_01
# CC2SC_CV10_TR_01 CC2SC_S20P_TR_01 SC2CC_CV10_TR_01 SC2CC_S20P_TR_01 -03.txt
# train_result.txt newdata_clean_noseg_seg

data_file = open(data_path, encoding='utf-8')
trained_file = open(trained_path, 'r', encoding='utf-8')
line_cnt = len(data_file.readlines())
tline_cnt = len(trained_file.readlines())

print(line_cnt, end="行\n")
print("-----")

input_texts = []
target_texts = []
#input_characters = set()
#target_characters = set()

ref_classical = []
ref_standard = []
can_trained = []

with open(data_path, 'r', encoding='utf-8') as f:
    lines = f.read().split('\n')
for line in lines[: min(line_cnt, len(lines) - 1)]:
    #
    input_text, target_text = line.split('\t')
    #print(input_text)
    # 不拆.append 拆.extend
    #input_text.replace('\ufeff', '')
    
    #文言與白話數目不同，新句無白話，尚未處理
    ref_classical.append(input_text.split())
    ref_standard.append(target_text.split())

with open(trained_path, 'r', encoding='utf-8') as f2:
    lines = f2.read().split('\n')
for line in lines[: min(tline_cnt, len(lines) - 1)]:
    can_trained.append(line.split(' '))

#print(ref) 檢視文言
#print("-----")
#print(ref_standard) #檢視白話答案
#print("-----")
#print(can_trained)  #檢視訓練結果

#測試滿分
#reference = ['我', '很', '喜歡', '他']
#candidate = ['我', '很', '喜歡', '他'] #93
##candidate = ['我', '非常', '他', '喜歡']
#reference = ['我', '很', '喜', '歡', '他']
#candidate = ['我', '很', '喜', '歡', '他'] #100
#score = sentence_bleu(reference, candidate)
#print('100分 %f' %score)
#reference = ['我', '很', '喜', '歡', '他']
#candidate = ['他', '喜', '歡', '很', '我'] #100
#score = sentence_bleu(reference, candidate)
#print('100分 %f' %score)
#reference = ['我', '很', '喜歡', '他']
#candidate = ['我', '很', '喜歡', '他'] #93
#score = sentence_bleu(reference, candidate, weights=(0.25, 0.25, 0.25, 0.25))
#print('100分 %f' %score)
    
# str.startswith(str, beg=0, end=len(str));
#print("----- -----")
cnt_line = len(ref_standard)
cnt_score = 0
#print(len(ref_standard))
#print(len(can_trained))
for i in range(0,len(ref_standard)-1):
#    print(ref_standard[i])
#    print(can_trained[i])
    if not ref_standard[i]:
#        print('Score: None')
        cnt_line -= 1
#        print("-----")
        continue
    score = sentence_bleu(ref_standard[i], can_trained[i])
#    print('Score: %f' % score)
    cnt_score += score
#    print("-----")
    #for j in range(0,len(ref_standard[i])):
    #    print(ref_standard[i])
    #    print("-----")
    #    print(can_trained[i])
        #score += sentence_bleu(ref_standard[i][j], can_trained[i][j])
        #print(score)
#print("BLEU Score")
bleu_score = cnt_score/cnt_line

##print(ref_standard[i])

#print('BLEU Score: %f / BLEU Rate: %.2f%%' % (bleu_score, bleu_score*100))
print('BLEU Score: %.2f' % (bleu_score*100))
#print("-----")

# BLEU N-gram Score
# Individual N-gram

# N-gram
weights = [1,1,1,1]
def ngram(reference,candidate):
    p_gram = [0,0,0,0]
    for i, w in enumerate(weights, start=0):
        p_i = modified_precision([reference], candidate, i+1)
        p_gram[i] = float(p_i.numerator) / float(p_i.denominator)
    #print(p_gram)
    return p_gram

cnt_line = len(ref_standard)
cnt_score = [0,0,0,0]
show_score = []
#for i in range(0,len(ref_standard)-1):
for i in range(0,len(ref_standard)):
    if not ref_standard[i]:
        cnt_line -= 1
        continue
    score = ngram(ref_standard[i], can_trained[i])
    for j in range(0,4):
        cnt_score[j] += score[j]
    #印完整數據
    #print(score)
    #print(score[3])
    
    #cnt_score += score
    #show_score.append(cnt_score)
for k in range(0,4):
    cnt_score[k] = cnt_score[k]*100/cnt_line
#print("---")

#印[11.560320615986623, 0.8581842279655401, 0.14910536779324055, 0]
#print(cnt_score)

#print('N-gram: %f' % (cnt_score*100/cnt_line))

g1 = cnt_score[0]
g2 = cnt_score[1]
g3 = cnt_score[2]
g4 = cnt_score[3]

print('1-gram: %f' % g1)
print('2-gram: %f' % g2)
print('3-gram: %f' % g3)
print('4-gram: %f' % g4)
'''
# BLEU N-gram Score
# 1-gram
cnt_line = len(ref_standard)
cnt_score = 0
for i in range(0,len(ref_standard)-1):
    #print(ref_standard[i])
    #print(can_trained[i])
    if not ref_standard[i]:
        #print('Score: None')
        cnt_line -= 1
        #print("-----")
        continue
    score = sentence_bleu(ref_standard[i], can_trained[i], weights=(1, 0, 0, 0))
    #print('Score: %f' % score)
    cnt_score += score
print('1-gram: %f' % (cnt_score*100/cnt_line))
#print("-----")

# 2-gram
cnt_line = len(ref_standard)
cnt_score = 0
for i in range(0,len(ref_standard)-1):
    #print(ref_standard[i])
    #print(can_trained[i])
    if not ref_standard[i]:
        #print('Score: None')
        cnt_line -= 1
        #print("-----")
        continue
    score = sentence_bleu(ref_standard[i], can_trained[i], weights=(0, 1, 0, 0))
    #score = sentence_bleu(ref_standard[i], can_trained[i], weights=(0.5, 0.5, 0, 0))
    #print('Score: %f' % score)
    cnt_score += score
print('2-gram: %f' % (cnt_score*100/cnt_line))
#print("-----")

# 3-gram
cnt_line = len(ref_standard)
cnt_score = 0
for i in range(0,len(ref_standard)-1):
    #print(ref_standard[i])
    #print(can_trained[i])
    if not ref_standard[i]:
        #print('Score: None')
        cnt_line -= 1
        #print("-----")
        continue
    score = sentence_bleu(ref_standard[i], can_trained[i], weights=(0, 0, 1, 0))
    #score = sentence_bleu(ref_standard[i], can_trained[i], weights=(0.33, 0.33, 0.33, 0))
    #print('Score: %f' % score)
    cnt_score += score
print('3-gram: %f' % (cnt_score*100/cnt_line))
#print("-----")

# 4-gram
cnt_line = len(ref_standard)
cnt_score = 0
for i in range(0,len(ref_standard)-1):
    #print(ref_standard[i])
    #print(can_trained[i])
    if not ref_standard[i]:
        #print('Score: None')
        cnt_line -= 1
        #print("-----")
        continue
    score = sentence_bleu(ref_standard[i], can_trained[i], weights=(0, 0, 0, 1))
    #score = sentence_bleu(ref_standard[i], can_trained[i], weights=(0.25, 0.25, 0.25, 0.25))
    #print('Score: %f' % score)
    cnt_score += score
print('4-gram: %f' % (cnt_score*100/cnt_line))
#print("-----")
'''

# BLEU Cumulative N-gram Score
#print('Cumulative 1-gram: %f' % sentence_bleu(ref_standard, can_trained, weights=(1, 0, 0, 0)))
cnt_line = len(ref_standard)
cnt_score = 0
for i in range(0,len(ref_standard)-1):
    #print(ref_standard[i])
    #print(can_trained[i])
    if not ref_standard[i]:
        #print('Score: None')
        cnt_line -= 1
        #print("-----")
        continue
    score = sentence_bleu(ref_standard[i], can_trained[i], weights=(1, 0, 0, 0))
    #print('Score: %f' % score)
    cnt_score += score
print('Cumulative 1-gram: %f' % (cnt_score*100/cnt_line))
#print("-----")

#print('Cumulative 2-gram: %f' % sentence_bleu(ref_standard, can_trained, weights=(0.5, 0.5, 0, 0)))
cnt_line = len(ref_standard)
cnt_score = 0
for i in range(0,len(ref_standard)-1):
    #print(ref_standard[i])
    #print(can_trained[i])
    if not ref_standard[i]:
        #print('Score: None')
        cnt_line -= 1
        #print("-----")
        continue
    score = sentence_bleu(ref_standard[i], can_trained[i], weights=(0.5, 0.5, 0, 0))
    #print('Score: %f' % score)
    cnt_score += score
print('Cumulative 2-gram: %f' % (cnt_score*100/cnt_line))
#print("-----")

#print('Cumulative 3-gram: %f' % sentence_bleu(ref_standard, can_trained, weights=(0.33, 0.33, 0.33, 0)))
cnt_line = len(ref_standard)
cnt_score = 0
for i in range(0,len(ref_standard)-1):
    #print(ref_standard[i])
    #print(can_trained[i])
    if not ref_standard[i]:
        #print('Score: None')
        cnt_line -= 1
        #print("-----")
        continue
    score = sentence_bleu(ref_standard[i], can_trained[i], weights=(0.33, 0.33, 0.33, 0))
    #print('Score: %f' % score)
    cnt_score += score
print('Cumulative 3-gram: %f' % (cnt_score*100/cnt_line))
#print("-----")

#print('Cumulative 4-gram: %f' % sentence_bleu(ref_standard, can_trained, weights=(0.25, 0.25, 0.25, 0.25)))
cnt_line = len(ref_standard)
cnt_score = 0
for i in range(0,len(ref_standard)-1):
    #print(ref_standard[i])
    #print(can_trained[i])
    if not ref_standard[i]:
        #print('Score: None')
        cnt_line -= 1
        #print("-----")
        continue
    score = sentence_bleu(ref_standard[i], can_trained[i], weights=(0.25, 0.25, 0.25, 0.25))
    #print('Score: %f' % score)
    cnt_score += score
print('Cumulative 4-gram: %f' % (cnt_score*100/cnt_line))
#print("-----")
