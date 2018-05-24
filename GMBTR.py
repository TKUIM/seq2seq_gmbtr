# Using the Sequence to Sequence Generative Model for Bidirectional Text Rewriting

from __future__ import print_function
from keras.models import Model
from keras.layers import Input, LSTM, Dense
from sklearn.model_selection import StratifiedKFold
import numpy as np
from sklearn.model_selection import train_test_split

# 64 -> 32
batch_size = 100  # 訓練時每批次樣本數 Batch size for training.
# 100 -> 150
# 200 overfitting
# 0.2 (150-200) / cv 0.1 (180-200)
epochs = 100  # 期數，訓練幾輪完整資料 Number of epochs to train for.
# 256 -> 128
latent_dim = 100  # 維度 Latent dimensionality of the encoding space.
# 10000 -> 1000
num_samples = 1000  # 訓練的樣本數 Number of samples to train on.
# 0.2
#validation_split = 0.2  # 自訓練資料集抽取的測試資料集比率
validation_cross_split =0.1  # 驗證比例
validation_fold = 10

# 資料集的路徑
data_path = 'D:/Paper/SCtoCC.txt'#'test/CCtoSC.txt'
# CCtoSC SCtoCC
# test/ CCtoSC_clean.txt SCtoCC_clean.txt

# Vectorize the data.
input_texts = []
target_texts = []
input_characters = set()
target_characters = set()
stop = set([' ', '\t'])
with open(data_path, 'r', encoding='utf-8-sig') as f:
    lines = f.read().split('\n')
for line in lines[: min(num_samples, len(lines))]:
    input_text, target_text = line.split('\t')
    # 以 Tab 分隔，前為輸入，後為輸出，以斷行 \n 分句。
    target_text = '\t' + target_text + '\n'
    input_texts.append(input_text)
    target_texts.append(target_text)
    for char in input_text:
        if char in stop:#== ' ': #or char == u'\ufeff'
            continue
        if char not in input_characters:
            input_characters.add(char)
    for char in target_text:
        if char in stop:
            continue
        if char not in target_characters:
            target_characters.add(char)
#print(lines)
print(input_characters)
print(target_characters)

# 處理相關數據
input_characters = sorted(list(input_characters))
target_characters = sorted(list(target_characters))
num_encoder_tokens = len(input_characters)
num_decoder_tokens = len(target_characters)
max_encoder_seq_length = max([len(txt) for txt in input_texts])
max_decoder_seq_length = max([len(txt) for txt in target_texts])

# 列出相關數據
print('Number of samples:', len(input_texts))
print('Number of unique input tokens:', num_encoder_tokens)
print('Number of unique output tokens:', num_decoder_tokens)
print('Max sequence length for inputs:', max_encoder_seq_length)
print('Max sequence length for outputs:', max_decoder_seq_length)


# One-hot向量
input_token_index = dict([(char, i) for i, char in enumerate(input_characters)])
target_token_index = dict([(char, i) for i, char in enumerate(target_characters)])

encoder_input_data = np.zeros(
    (len(input_texts), max_encoder_seq_length, num_encoder_tokens), dtype='float32')
decoder_input_data = np.zeros(
    (len(input_texts), max_decoder_seq_length, num_decoder_tokens), dtype='float32')
decoder_target_data = np.zeros(
    (len(input_texts), max_decoder_seq_length, num_decoder_tokens), dtype='float32')

for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):
    enIndex = 0
    for t, char in enumerate(input_text):
        if char == ' ': continue
        encoder_input_data[i, enIndex, input_token_index[char]] = 1.
        enIndex = enIndex + 1
    
    deIndex = 0
    for t, char in enumerate(target_text):
        # decoder_target_data is ahead of decoder_input_data by one timestep
        if char in stop: continue
        decoder_input_data[i, deIndex, target_token_index[char]] = 1.
        deIndex = deIndex + 1
        
        #decoder_input_data[i, t, target_token_index[char]] = 1.
        if deIndex > 0:
            # decoder_target_data will be ahead by one timestep
            # and will not include the start character.
            decoder_target_data[i, deIndex - 1, target_token_index[char]] = 1.


# Define an input sequence and process it.
encoder_inputs = Input(shape=(None, num_encoder_tokens))
encoder = LSTM(latent_dim, return_state=True)
encoder_outputs, state_h, state_c = encoder(encoder_inputs)
# We discard `encoder_outputs` and only keep the states.
encoder_states = [state_h, state_c]

# Set up the decoder, using `encoder_states` as initial state.
decoder_inputs = Input(shape=(None, num_decoder_tokens))
# We set up our decoder to return full output sequences,
# and to return internal states as well. We don't use the
# return states in the training model, but we will use them in inference.
decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
decoder_dense = Dense(num_decoder_tokens, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

# Define the model that will turn
# `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

# 跑訓練
# optimizer='rmsprop' optimizer='adam'
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
'''
train_history=model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
          batch_size=batch_size,
          epochs=epochs,
          validation_split=validation_split)
'''
# (交叉)驗證
#train_history=model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
#        epochs=epochs,
#        validation_split=validation_cross_split)

'''
def split_fold(s, n):
    fn = len(s)//n
    rn = len(s)%n
    ar = [fn+1]*rn+ [fn]*(n-rn)
    si = [i*(fn+1) if i<rn else (rn*(fn+1)+(i-rn)*fn) for i in xrange(n)]
    sr = [s[si[i]:si[i]+ar[i]] for i in xrange(n)]
    return sr
#print split_fold(input_texts,validation_fold)
'''

# fix random seed for reproducibility
# 用 seed 確保再現性
seed = 16313
#numpy.random.seed(seed)

#cvscores = []
#train_test_split(y,shuffle=true,test_size=0.1)


entrain, entest, detrain, detest, target_train, target_test = train_test_split(encoder_input_data, 
                                                                               decoder_input_data, 
                                                                               decoder_target_data, test_size=0.1)

train_history=model.fit([entrain, detrain], target_train, 
                        epochs=epochs, batch_size=batch_size, verbose=1)


#    train_history=model.fit(encoder_input_data[train], decoder_input_data[train], 
#                            epochs=epochs, batch_size=batch_size, verbose=0)
#scores = model.evaluate([entest, detest], verbose=1)
#print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
#cvscores.append(scores[1] * 100)

# 儲存模型
model.save('s2s.h5')

# Define sampling models
encoder_model = Model(encoder_inputs, encoder_states)

decoder_state_input_h = Input(shape=(latent_dim,))
decoder_state_input_c = Input(shape=(latent_dim,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
decoder_outputs, state_h, state_c = decoder_lstm(
    decoder_inputs, initial_state=decoder_states_inputs)
decoder_states = [state_h, state_c]
decoder_outputs = decoder_dense(decoder_outputs)
decoder_model = Model(
    [decoder_inputs] + decoder_states_inputs,
    [decoder_outputs] + decoder_states)

# Reverse-lookup token index to decode sequences back to
# something readable.
reverse_input_char_index = dict((i, char) for char, i in input_token_index.items())
reverse_target_char_index = dict((i, char) for char, i in target_token_index.items())

def decode_sequence(input_seq):
    # Encode the input as state vectors.
    states_value = encoder_model.predict(input_seq)

    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1, 1, num_decoder_tokens))
    # Populate the first character of target sequence with the start character.
#    target_seq[0, 0, target_token_index['\t']] = 1.

    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :]) #####
        sampled_char = reverse_target_char_index[sampled_token_index]
        decoded_sentence += sampled_char

        # Exit condition: either hit max length
        # or find stop character.
        if (sampled_char == '\n' or
           len(decoded_sentence) > max_decoder_seq_length):
            stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1, 1, num_decoder_tokens))
        target_seq[0, 0, sampled_token_index] = 1.

        # Update states
        states_value = [h, c]

    return decoded_sentence

'''
sys.setrecursionlimit(1000000)
# set parameters:
vocab_dim = 100# word2vec size ：訓練出的詞向量會有幾維
maxlen = 100
n_iterations = 1  # ideally more..
n_exposures = 10
window_size = 5
batch_size = 32
# n_epoch = 500
n_epoch = 100
# n_epoch = 5
input_length = 100
cpu_count = multiprocessing.cpu_count()
'''

#建立詞典，Return每個詞的索引、詞向量，以及每個語句所對應的詞語索引。
def create_dictionaries(model=None,combined=None):
    if(combined is not None)and(model is not None):
        gensim_dict = Dictionary()
        gensim_dict.doc2bow(model.wv.vocab.keys(), allow_update=True)
        w2indx = {v: k+1 for k, v in gensim_dict.items()} #所有詞頻 >10 的詞語索引
        w2vec = {word: model[word] for word in w2indx.keys()} #所有詞頻 >10 的詞語詞向量

        def parse_dataset(combined):
            # Words become integers
            data=[]
            for sentence in combined:
                new_txt = []
                for word in sentence:
                    try:
                        new_txt.append(w2indx[word])
                    except:
                        new_txt.append(0)
                data.append(new_txt)
            return data
        combined=parse_dataset(combined)
        combined= sequence.pad_sequences(combined, maxlen=maxlen) #每個語句所含詞語對應的索引，句中含有詞頻 <10 的詞語，索引為0
        return w2indx, w2vec,combined
    else:
        print ('No data provided...')

#建立詞典，Return每個詞的索引、詞向量，以及每個語句所對應的詞語索引。
def word2vec_train(combined):
    model = Word2Vec(size=vocab_dim,
                     min_count=n_exposures,
                     window=window_size,
                     workers=cpu_count,
                     iter=n_iterations)
    model.build_vocab(combined)
    model.train(combined)
    model.save('Word2vec_model.pkl')
    index_dict, word_vectors,combined = create_dictionaries(model=model,combined=combined)
    return   index_dict, word_vectors,combined

def get_data(index_dict,word_vectors,combined,y):
    n_symbols = len(index_dict) + 1  #所有詞語的索引數，詞頻 <10 的詞語索引為 0，所以數值 +1。
    embedding_weights = np.zeros((n_symbols, vocab_dim)) #索引為0的詞語，詞向量全為 0。
    for word, index in index_dict.items(): #從索引為 1 的詞語開始，對每個詞語對應其詞向量。
        embedding_weights[index, :] = word_vectors[word]
    x_train, x_test, y_train, y_test = train_test_split(combined, y, test_size=0.1)
    print (x_train.shape,y_train.shape)
    return n_symbols,embedding_weights,x_train,y_train,x_test,y_test

###
print(encoder_input_data.shape)
print(decoder_input_data.shape)

print(entrain.shape, entest.shape, detrain.shape, detest.shape)
print(target_train.shape)
###

# 前指定筆數
#for i in range(len(LIST_NAME)):
#for i, LIST_NAME in enumerate(LIST_NAME):
for seq_index in range(10): #對新測試集重做
    # Take one sequence (part of the training set)
    # for trying out decoding.
    input_seq = encoder_input_data[seq_index: seq_index + 1]#seq_index: seq_index + 1
    decoded_sentence = decode_sequence(input_seq)
    #print('-')
    #print('Input sentence:', input_texts[seq_index].replace(' ', ''))
    #print('Decoded sentence:', decoded_sentence.replace(' ', ''))
    
    print('Input sentence:', input_texts[seq_index])
    print('Decoded sentence:', decoded_sentence)
###

# 隨機挑選
import random
for seq_index in range(10):
    ran = random.randrange(0, len(input_texts))
    input_seq = encoder_input_data[ran: ran+1]
    decoded_sentence = decode_sequence(input_seq)
    #print('-')
    print('Input sentence:', input_texts[ran].replace(' ', ''))
    print('Decoded sentence:', decoded_sentence.replace(' ', ''))
###

# 指定輸入
#while True:
try:
    input_seq = decode_sequence(input())
    #query = input_texts.index(input())
    #input_seq = encoder_input_data[query: query+1]
        #print(input_seq)
        #input_seq = input()
    decoded_sentence = decode_sequence(input_seq)
    print('Input sentence:', input_texts[query].replace(' ', ''))
    print('Decoded sentence:', decoded_sentence.replace(' ', ''))
except Exception as e:
    print(repr(e))
###

# 檢視資料
'''
r    讀取(檔案需存在)
w    新建檔案寫入(檔案可不存在，若存在則清空)
a    資料附加到舊檔案後面(游標指在EOF)
r+   讀取舊資料並寫入(檔案需存在且游標指在開頭)
w+   清空檔案內容，新寫入的東西可在讀出(檔案可不存在，會自行新增)
a+   資料附加到舊檔案後面(游標指在EOF)，可讀取資料
b    二進位模式
'''

# f = open('newdata_clean.txt', 'r', encoding='utf-8')
# 指定輸出全檔 筆數為行數
#for i in range(len(LIST_NAME)):
#for i, LIST_NAME in enumerate(LIST_NAME):
#for seq_index in range(10):
for seq_index in range(len(encoder_input_data)):
    input_seq = encoder_input_data[seq_index: seq_index + 1]
    decoded_sentence = decode_sequence(input_seq)
    #print('-')
    print('Input sentence:', input_texts[seq_index].replace(' ', ''))
    print('Decoded sentence:', decoded_sentence.replace(' ', ''))
###

# 輸出訓練結果
import re

#data_path = 'newdata_clean.txt'
data_file = open(data_path, encoding='utf-8') 
line_cnt = len(data_file.readlines())
input_texts = []
target_texts = []
ref_classical = []
can_trained = []

with open('test/train_result.txt', 'w', encoding='utf-8') as f:
    for seq_index in range(len(encoder_input_data)):
        input_seq = encoder_input_data[seq_index: seq_index + 1]
        decoded_sentence = decode_sequence(input_seq)
        decoded_sentence = re.sub(' +', ' ', decoded_sentence)
        decoded_sentence = re.sub('^ ', '', decoded_sentence)
        decoded_sentence = re.sub(' $', '', decoded_sentence)
        #decoded_sentence = re.sub(' \n', '', decoded_sentence)
        f.write(decoded_sentence)
        #ref_classical.append(input_texts[seq_index])
        can_trained.append(decoded_sentence)
        f.flush()
    #print(ref_classical)
    print(can_trained)
###
%matplotlib inline
import matplotlib.pyplot as plt
#print(train_history)
def show_train_history(train_history,train,validation):
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.title('Train History')
    plt.ylabel(train)
    plt.xlabel('Epoch')
    plt.legend(['train','validation'],loc='upper left')
    plt.show()
#plt.savefig("filename.png",dpi=300,format="png")
###
show_train_history(train_history,'acc','val_acc')
###
