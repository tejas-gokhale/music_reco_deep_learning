**06-APR-2017**
--
Resolved dataset download problems
Figured out a way to convert .h5 to csv
Only 77% of the songs occur both in the features data and triplets data
So triplets data was pruned

Can now work on all 3 methods using this.
1. features.csv for Content-Based
2. triplets.csv for Collaborative
3. both for Deep

Work to be done:
1. @Deepika: feature processing (normalization etc) and network architecture
2. @Raghav: content based recommendation
3. @Tejas: Collaborative Filtering

Tejas: 
'''
I am reading my NLP slides that talk about efficient similarity measurement. We could potentially use that for our purposes.
'''
