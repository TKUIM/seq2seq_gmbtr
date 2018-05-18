import logging

from gensim.models import word2vec

def main():

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    sentences = word2vec.LineSentence("wiki_sc_seg.txt")
    model = word2vec.Word2Vec(sentences, size=100)
    # class gensim.models.word2vec.Word2Vec
    # (sentences=None, size=100, alpha=0.025, window=5, min_count=5, max_vocab_size=None, 
    # sample=0.001, seed=1, workers=3, min_alpha=0.0001, sg=0, ### sg={1,0} 1 skip-gram; 0 CBOW.
    # hs=0, negative=5, ### hs={1,0} 1 hierarchical softmax / 0 and negative is non-zero, negative sampling
    # cbow_mean=1, hashfxn=<built-in function hash>, iter=5, null_word=0, 
    # trim_rule=None, sorted_vocab=1, batch_words=10000)

    #儲存模型
    model.save("wikisc_w2v.model")

    #讀取模型
    # model = word2vec.Word2Vec.load("model_name")

if __name__ == "__main__":
    main()
