import random

import numpy as np

prices = [random.randint(1, 10000) for _ in range(10000)]
extreme_prices = [random.randint(10000, 1000000) for _ in range(10)]
for el in extreme_prices:
    prices.append(el)

# Sort the prices
prices.sort()

# Calculate the price difference between adjacent products
price_diffs = [prices[i+1] - prices[i] for i in range(len(prices)-1)]

# Define a threshold for price difference
threshold = np.percentile(prices, 99)

# Filter out the products where the price difference is below the threshold
selected_products = [(prices[i], prices[i+1]) for i in range(len(prices)-1) if price_diffs[i] >= threshold]

# Print the selected products
print(selected_products)
