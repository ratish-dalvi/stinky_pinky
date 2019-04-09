import pyphen
import gensim.models
import pronouncing as pron

dic = pyphen.Pyphen(lang='en')

# Download and a trained word2vec model from `http://vectors.nlpl.eu/explore/embeddings/en/models/`
w2v = gensim.models.KeyedVectors.load_word2vec_format("./model.txt", binary=False)


# Set the details of the question here:
# Question: Fresh water favorite served in the name of a higher power.
w_list1 = w2v.most_similar(positive=["god_NOUN", "religion_NOUN"], topn=1000)
w_list2 = w2v.most_similar(positive=["fish_NOUN", "freshwater_NOUN"], topn=1000)


def extract_words(w_list):
    words_new = []
    for token in w_list:
        distance = 1 - token[1]
        for w in token[0].split("_")[0].split("::"):
            n_syllables = len(dic.inserted(w).split('-'))
            words_new.append((w, distance, n_syllables))
    return words_new

w_list1 = extract_words(w_list1)
w_list2 = extract_words(w_list2)

pairs = {}
for w1, sc1, ns1 in w_list1:
    rhymes = pron.rhymes(w1)
    for w2, sc2, ns2 in w_list2:
        if w2 in rhymes:
            len_gap = abs(len(w1) - len(w2))
            syllable_gap = abs(ns1-ns2)
            score = sc1 * sc2
            key = "%s-%s" % (w1.lower(), w2.lower())
            key_rev = "%s-%s" % (w2.lower(), w1.lower())
            if key in pairs:
                score = max(score, pairs[key][0])
                del pairs[key]
            if key_rev in pairs:
                score = max(score, pairs[key_rev][0])
                del pairs[key_rev]
            pairs[key] = (score, syllable_gap, len_gap)


def wise_sort(x):
    key, values = x
    distance, syllable_gap, len_gap = values
    return syllable_gap, len_gap, round(distance, 1)

# Print the output
cnt = 0
for key, _ in sorted(pairs.items(), key=wise_sort):
    print(cnt, key)
    cnt += 1
