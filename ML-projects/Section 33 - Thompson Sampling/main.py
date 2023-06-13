import pandas as pd
import random
import matplotlib.pyplot as plt

dataset = pd.read_csv('Ads_CTR_Optimisation.csv')
num_ads = dataset.shape[1]
num_customers = dataset.shape[0]
ads_selected = []
number_of_reward1 = [0] * num_ads
number_of_reward0 = [0] * num_ads
total_reward = 0

for n in range(num_customers):
    ad = 0
    max_random = 0
    for i in range(num_ads):
        random_beta = random.betavariate(number_of_reward1[i] + 1, number_of_reward0[i] + 1)
        if random_beta > max_random:
            max_random = random_beta
            ad = i
    ads_selected.append(ad)
    reward = dataset.values[n, ad]
    if reward == 1:
        number_of_reward1[ad] += 1
    else:
        number_of_reward0[ad] += 1
    total_reward += reward

print(total_reward)

plt.hist(ads_selected)
plt.title("Histogram of ads selections")
plt.xlabel("Ads")
plt.ylabel("Number of times each ad was selected")
plt.show()
