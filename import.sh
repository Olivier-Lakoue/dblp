export DATA_DIR=/home/markhneedham/projects/dblp/data

./bin/neo4j-admin import \
  --database=foo.db \
  --nodes:Author=${DATA_DIR}/authors_header.csv,${DATA_DIR}/authors.csv \
  --nodes:Article=${DATA_DIR}/articles_header.csv,/${DATA_DIR}/articles.csv \
  --relationships:REFERENCES=${DATA_DIR}/article_REFERENCES_article_header.csv,${DATA_DIR}/article_REFERENCES_article.csv \
  --relationships:AUTHOR=${DATA_DIR}/article_AUTHOR_author_header.csv,${DATA_DIR}/article_AUTHOR_author.csv \
  --multiline-fields=true
