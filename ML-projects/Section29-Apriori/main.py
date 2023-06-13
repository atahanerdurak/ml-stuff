import pandas as pd
from apyori import apriori, load_transactions


# take results from apriori and put it in a good-looking dataframe
def inspect(results):
    lhs = [tuple(result[2][0][0])[0] for result in results]
    rhs = [tuple(result[2][0][1])[0] for result in results]
    supports = [result[1] for result in results]
    confidences = [result[2][0][2] for result in results]
    lifts = [result[2][0][3] for result in results]
    return list(zip(lhs, rhs, supports, confidences, lifts))


dataset = pd.read_csv("Market_Basket_Optimisation.csv", header=None)
transactions = []
for i in range(dataset.shape[0]):
    transactions.append([str(dataset.values[i, j]) for j in range(dataset.shape[1])])

rules = apriori(transactions=transactions, min_support=3 * 7 / dataset.shape[0], min_confidence=0.2,
                min_lift=3, min_length=2, max_length=2)
market_results = list(rules)

# we can do above with apyori (no need to create transactions etc.)
# with open('Market_Basket_Optimisation.csv') as f:
#     transactions = apyori.load_transactions(f, delimiter=',')
#     results = list(apyori.apriori(transactions=transactions, min_support=3*7/dataset.shape[0], min_confidence=0.2,
#                 min_lift=3, min_length=2, max_length=2))

resultsinDataFrame = pd.DataFrame(inspect(market_results),
                                  columns=['Left Hand Side', 'Right Hand Side', 'Support', 'Confidence', 'Lift'])

print(resultsinDataFrame.nlargest(n=20, columns='Lift'))
