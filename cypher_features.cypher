CALL algo.louvain.stream("Author", "CO_AUTHOR_EARLY", {includeIntermediateCommunities:true})
YIELD nodeId, community, communities
WITH algo.getNodeById(nodeId) AS node, communities[0] AS smallestCommunity
SET node.louvainTrain = smallestCommunity;


CALL algo.louvain.stream("Author", "CO_AUTHOR", {includeIntermediateCommunities:true})
YIELD nodeId, community, communities
WITH algo.getNodeById(nodeId) AS node, communities[0] AS smallestCommunity
SET node.louvainTest = smallestCommunity;


CALL algo.triangleCount('Author', 'CO_AUTHOR_EARLY', { write:true,
  writeProperty:'trianglesTrain', clusteringCoefficientProperty:'coefficientTrain'});

CALL algo.triangleCount('Author', 'CO_AUTHOR', { write:true,
  writeProperty:'trianglesTest', clusteringCoefficientProperty:'coefficientTest'});


CALL algo.pageRank("Author", "CO_AUTHOR_EARLY", {writeProperty: "pagerankTrain"});

CALL algo.pageRank("Author", "CO_AUTHOR", {writeProperty: "pagerankTest"});


CALL algo.labelPropagation("Author", "CO_AUTHOR_EARLY", "BOTH", {partitionProperty: "partitionTrain"});


CALL algo.labelPropagation("Author", "CO_AUTHOR", "BOTH", {partitionProperty: "partitionTest"});