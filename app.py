from pyspark.sql import SparkSession
from pyspark.sql.functions import avg, count, desc

# Initialize SparkSession
spark = SparkSession.builder.appName("NFT Analysis").getOrCreate()

# Load data from a file or any other data source
data = spark.read.csv("path/to/nft_data.csv", header=True, inferSchema=True)

# Calculate average number of NFTs per user
avg_nfts_per_user = data.groupBy("user_id").count().agg(avg("count")).first()[0]

# Calculate distribution of user levels
user_levels = data.groupBy("user_level").count().orderBy("user_level")

# Calculate top 10 most popular NFT types
top_nft_types = data.groupBy("nft_type").count().orderBy(desc("count")).limit(10)

# Print the results
print("Average number of NFTs per user: ", avg_nfts_per_user)
print("Distribution of user levels:")
user_levels.show()
print("Top 10 most popular NFT types:")
top_nft_types.show()

# Stop SparkSession
spark.stop()
