import json
import sys
from collections import defaultdict
def read():
    filepath = sys.argv[1]
    with open(filepath, encoding="utf-8") as f:
        squad = json.load(f)
        for article in squad["data"]:
            title = article.get("title", "")
            suffix = ''
            if 'type' in article:
                suffix = article['type']
            for paragraph in article["paragraphs"]:
                context = paragraph["context"]  # do not strip leading blank spaces GH-2585
                for qa in paragraph["qas"]:
                    answer_starts = [answer["answer_start"] for answer in qa["answers"]]
                    answers = [answer["text"] for answer in qa["answers"]]
                    # Features currently used are "context", "question", and "answers".
                    # Others are extracted here for the ease of future expansions.
                    print(qa["id"], qa["question"], answers, sep='\t')

def get_pos_negs():
    query2para = defaultdict(list)
    for line in sys.stdin:
        items = line.strip().split('\t')
        if len(items) != 5:
            continue
        query2para[items[0]+'\x01' + items[1]].append(items[2:])
    good, bad = 0, 0
    paraset = set()
    for k, v in query2para.items():
        minpos = 1000
        outres = []
        poslist = []
        for tup in v:
            score = float(tup[1])
            if tup[2] == "1":
                poslist.append(tup)
                if score < minpos:
                    minpos = score
            else:
                if score > minpos:
                    outres.append(tup)
        outres.sort(key=lambda x:float(x[1]))
        for pos in poslist:
            posscore = float(pos[1])
            for neg in outres:
                negscore = float(neg[1])
                if -5 > negscore - posscore and neg[0] not in paraset:
                    paraset.add(neg[0])
                    print(k, pos[0], neg[0], pos[1], neg[1],sep='\t')
                    break
def read_file(filename):
    stopset = set()
    for line in open(filename):
        stopset.add(line.strip())
    return stopset

def read_du():
    import jieba
    stopset = read_file('./stopwords.txt')
    for line in sys.stdin:
        items = line.strip().split('\t')
        query = items[0].split('\x01')[0]
        query_tokens = [a for a in jieba.cut(query) if a not in stopset]
        inpos, inneg = 0, 0
        for w in query_tokens:
            if w in items[1]:
                inpos += 1
            if w in items[2]:
                inneg += 1
        if inneg < inpos:
            print(line.strip())

def convert_line():
    for line in sys.stdin:
        items = line.strip().split('\t')
        query = items[0].split('\x01')[1]
        print(query, items[1], items[2], sep='\t')

if __name__ == "__main__":
    convert_line()


