import json
import sys

if __name__ == "__main__":
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
