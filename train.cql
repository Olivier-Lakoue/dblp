// Pick out 10,000 authors who published in 2017 as our training examples
CALL db.index.explicit.seekRelationships("coauthor", "year", $year)
YIELD relationship
WITH startNode(relationship) AS s, endNode(relationship) AS e, relationship
WITH collect([e, s]) AS authors
WITH apoc.coll.toSet(apoc.coll.flatten(authors)) AS uniqueAuthors
UNWIND uniqueAuthors as author
WITH author, size([(author)<-[:AUTHOR]-(article) WHERE article.year < $year | article]) as size
ORDER BY size DESC
LIMIT 10000
SET author:Train;

MATCH (author:Author:Train)
WHERE rand() < 0.01
SET author:SmallTrain;

// Potential neighbours
MATCH (author:Author:Train)
WITH author LIMIT 100
MATCH (author)-[:CO_AUTHOR*2]-(potentialCoAuthor:Train:Author)
WITH DISTINCT author, potentialCoAuthor
WHERE not((author)-[:CO_AUTHOR]-(potentialCoAuthor))
RETURN count(*);

// How many pairs in our training sets?
CALL db.index.explicit.seekRelationships("coauthor", "year", $year)
YIELD relationship
WITH startNode(relationship) AS s, endNode(relationship) AS e, relationship
WHERE s:SmallTrain and e:SmallTrain
RETURN count(*);
// 12
