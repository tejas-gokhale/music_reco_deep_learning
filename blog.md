**06-APR-2017**
--
Resolved dataset download problems
Figured out a way to convert .h5 to csv ...
Only 77% of the songs occur both in the features data and triplets data.
So triplets data was pruned.

Can now work on all 3 methods using this.
1. features.csv for Content-Based
2. triplets.csv for Collaborative
3. both for Deep

Work to be done:
1. @Deepika: feature processing (normalization etc) and network architecture
2. @Raghav: content based recommendation
3. @Tejas: Collaborative Filtering

Tejas: 
```
I am reading my NLP slides that talk about efficient similarity measurement.
We could potentially use that for our purposes.
```

**19-APR-2017**
--
Content-based recommendation code working.
Results stored in ./results

Decided to use lyrics as inputs to the deep-network. Needs formatting and linking MSD Song-id to Musixmatch Track-id.
Collaborative filtering in pipeline. top-n similar users using Pearson correlation found out. Working on Recommendations now.

Agenda for 20th:
1. Complete Project Progress Report
2. Formulate Deep Network Architecture

**04-MAY-2017**
- Data extracted from Bag-of-Words and stored in csv file after many strenuous attempts of a rather boring task.
- Data split as train-test-valid

**08-MAY-2017**
Three configurations:
1. input = top-100 words
2. input = all words
3. input = embeddings obtained from Auto-encoder/ tSNE

We aim to compare the outputs of these three in the report.

To-do:
1. Network architecture
2. Recommendation system
3. Comparisons of 3 approaches
4. Specific examples to show the predictions
5. Report
6. Video
4. 
