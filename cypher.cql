CALL apoc.periodic.iterate(
  'UNWIND ["dblp-ref-0.json","dblp-ref-1.json","dblp-ref-2.json","dblp-ref-3.json"] as file
   CALL apoc.load.json("file:///home/markhneedham/projects/dblp/dblp-ref/" + file)
   yield value
   WHERE value.venue = "Lecture Notes in Computer Science"
   return value',
  'MERGE (a:Article{index:value.id})
   ON CREATE SET a += apoc.map.clean(value,["id","authors","references"],[0])
   WITH a,value.authors as authors
   UNWIND authors as author
   MERGE (b:Author{name:author})
   MERGE (b)<-[:AUTHOR]-(a)'
,{batchSize: 10000, iterateList: true});


// Remove single article authors
match (a:Author)<-[r:AUTHOR]-(art:Article {venue: "Lecture Notes in Computer Science"})
WITH a, collect(r) as rels, count(*) AS count
WHERE count = 1
DELETE rels[0];

// Authors frequency
MATCH (a:Article)
WITH a, size((a)-[:AUTHOR]->()) AS authors
RETURN authors, count(*)
ORDER BY authors DESC
LIMIT 100;

// Co-authors
MATCH (a1)<-[:AUTHOR]-(paper)-[:AUTHOR]->(a2:Author)
WHERE paper.year < 2007
WITH a1, a2, paper
ORDER BY a1, paper.year
RETURN a1, a2, collect(paper)[0].year AS year, count(*) AS collaborations
ORDER BY collaborations DESC
LIMIT 10;


CALL apoc.periodic.iterate(
 'MATCH (a1:Author)
  RETURN a1',
 'MATCH (a1)<-[:AUTHOR]-(paper)-[:AUTHOR]->(a2:Author)
  WITH a1, a2, paper
  ORDER BY a1, paper.year
  WITH a1, a2, collect(paper)[0].year AS year, count(*) AS collaborations
  MERGE (a1)-[coauthor:CO_AUTHOR {year: year}]-(a2)
  SET coauthor.collaborations = collaborations',
  {});

CALL apoc.periodic.iterate(
 'MATCH (a1:Author)
  RETURN a1',
 'MATCH (a1)<-[:AUTHOR]-(paper)-[:AUTHOR]->(a2:Author)
  WITH a1, a2, paper
  ORDER BY a1, paper.year
  WITH a1, a2, collect(paper)[0].year AS year, count(*) AS collaborations
  WHERE year < 2006
  MERGE (a1)-[coauthor:CO_AUTHOR_EARLY {year: year}]-(a2)
  SET coauthor.collaborations = collaborations',
  {});

CALL apoc.periodic.iterate(
 'MATCH (a1:Author)
  RETURN a1',
 'MATCH (a1)<-[:AUTHOR]-(paper)-[:AUTHOR]->(a2:Author)
  WITH a1, a2, paper
  ORDER BY a1, paper.year
  WITH a1, a2, collect(paper)[0].year AS year, count(*) AS collaborations
  WHERE year >= 2006
  MERGE (a1)-[coauthor:CO_AUTHOR_LATE {year: year}]-(a2)
  SET coauthor.collaborations = collaborations',
  {});

CALL db.index.explicit.forRelationships("coauthor");

call apoc.periodic.iterate(
 'match (a1)-[coa:CO_AUTHOR]->(a2) RETURN coa',
 'CALL db.index.explicit.addRelationship("coauthor", coa, "year", coa.year)
  YIELD success
  return count(*)', {})

CALL db.index.explicit.seekRelationships("coauthor", "year", 2017)
YIELD relationship
RETURN startNode(relationship), endNode(relationship)
LIMIT 10;


// Finding links in 2017
CALL db.index.explicit.seekRelationships("coauthor", "year", $year)
YIELD relationship
WITH startNode(relationship) AS s, endNode(relationship) AS e, relationship

return s.name, e.name,
       size([(s)-[coa:CO_AUTHOR]-() WHERE coa.year < $year | coa]) AS author1Size,
       size([(e)-[coa:CO_AUTHOR]-() WHERE coa.year < $year | coa]) AS author2Size,
       size([(s)-[coa1:CO_AUTHOR]-()-[coa2:CO_AUTHOR]-(e) WHERE coa1.year < $year AND coa2.year < $year | coa1]) AS overlapSize
order by author1Size + author2Size DESC
LIMIT 100;

CALL db.index.explicit.seekRelationships("coauthor", "year", $year)
YIELD relationship
WITH startNode(relationship) AS s, endNode(relationship) AS e, relationship
WITH collect([e, s]) AS authors
WITH apoc.coll.toSet(apoc.coll.flatten(authors)) AS uniqueAuthors
UNWIND uniqueAuthors as author
return count(*);


// Runs for ages
CALL db.index.explicit.seekRelationships("coauthor", "year", $year)
YIELD relationship
WITH startNode(relationship) AS s, endNode(relationship) AS e, relationship
WITH s, e,
       size([(s)-[coa:CO_AUTHOR]-() WHERE coa.year < $year | coa]) AS author1Size,
       size([(e)-[coa:CO_AUTHOR]-() WHERE coa.year < $year | coa]) AS author2Size,
       size([(s)-[coa1:CO_AUTHOR]-()-[coa2:CO_AUTHOR]-(e) WHERE coa1.year < $year AND coa2.year < $year | coa1]) AS overlapSize,
       algo.similarity.jaccard([(s)-[coa:CO_AUTHOR]-(co) WHERE coa.year < $year | id(co)], [(e)-[coa:CO_AUTHOR]-(co) WHERE coa.year < 2017 | id(co)]) AS jaccard
WHERE overlapSize > 0
RETURN s.name, e.name, author1Size, author2Size, overlapSize, jaccard
order by jaccard DESC
LIMIT 1000;
