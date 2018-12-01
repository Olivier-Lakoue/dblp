// Find missing links for the training graph
MATCH (author:Author)
WHERE (author)-[:CO_AUTHOR_EARLY]-()
MATCH (author)-[:CO_AUTHOR_EARLY*2]-(other)
WHERE not((author)-[:CO_AUTHOR_EARLY]-(other))
RETURN id(author), id(other)
LIMIT 10;

// Find missing links for the test graph
MATCH (author:Author)
WHERE (author)-[:CO_AUTHOR_LATE]-()
MATCH (author)-[:CO_AUTHOR*2]-(other)
WHERE not((author)-[:CO_AUTHOR]-(other))
RETURN id(author), id(other)
LIMIT 10;
