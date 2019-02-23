import glob
import json
import csv

articles = {}
authors = set()

def write_header(file, fields):
    csv.writer(file, delimiter=",").writerow(fields)

with open("data/article_REFERENCES_article.csv", "w") as article_references_article_file, \
     open("data/article_REFERENCES_article_header.csv", "w") as article_references_article_header_file, \
     open("data/article_AUTHOR_author.csv", "w") as article_author_author_file, \
     open("data/article_AUTHOR_author_header.csv", "w") as article_author_author_header_file:
    write_header(article_references_article_header_file, [":START_ID(Article)", ":END_ID(Article)"])
    write_header(article_author_author_header_file, [":START_ID(Article)", ":END_ID(Author)"])

    articles_references_article_writer = csv.writer(article_references_article_file, delimiter=",")
    article_author_author_file_writer = csv.writer(article_author_author_file, delimiter=",")
    for file_path in glob.glob("dblp-ref/*.json"):
        with open(file_path, "r") as file:
            line = file.readline()
            while line:
                item = json.loads(line)
                articles[item["id"]] = { "abstract": item.get("abstract", ""), "title": item["title"]}

                for reference in item.get("references", []):
                    articles_references_article_writer.writerow([item["id"], reference])

                for author in item.get("authors", []):
                    authors.add(author)
                    article_author_author_file_writer.writerow([item["id"], author])

                line = file.readline()

with open("data/articles.csv", "w") as articles_file, \
     open("data/articles_header.csv", "w") as articles_header_file, \
     open("data/authors.csv", "w") as authors_file, \
     open("data/authors_header.csv", "w") as authors_header_file:
    write_header(articles_header_file, ["id:ID(Article)", "title:string", "abstract:string", "year:int"])
    write_header(authors_header_file, ["name:ID(Author)", "name:string"])

    articles_writer = csv.writer(articles_file, delimiter=",")
    for article_id in articles:
        article = articles[article_id]
        articles_writer.writerow([article_id, article["title"], article["abstract"], article.get("year")])

    authors_writer = csv.writer(authors_file, delimiter=",")
    for author in authors:
        authors_writer.writerow([author, author])
