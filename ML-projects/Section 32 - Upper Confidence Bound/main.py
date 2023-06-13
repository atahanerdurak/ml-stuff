import pandas as pd
import math
import matplotlib.pyplot as plt

dataset = pd.read_csv('Ads_CTR_Optimisation.csv')
num_ads = dataset.shape[1]
num_customers = dataset.shape[0]
Ni = [0] * num_ads
Ri = [0] * num_ads
total_reward = 0
ads_selected = []

for n in range(num_customers):
    ad = 0
    max_upper_bound = 0
    for i in range(num_ads):
        if Ni[i] > 0:
            average_reward = Ri[i] / Ni[i]
            delta_i = math.sqrt(3/2 * math.log(n+1) / Ni[i])
            upper_bound = average_reward + delta_i
        else:
            upper_bound = 1e400
        if upper_bound > max_upper_bound:
            max_upper_bound = upper_bound
            ad = i
    ads_selected.append(ad)
    Ni[ad] += 1
    reward = dataset.values[n, ad]
    Ri[ad] += reward
    total_reward += reward

print(Ni)
print(Ri)
print(total_reward)
print(ads_selected)

plt.hist(ads_selected)
plt.title("Histogram of ads selections")
plt.xlabel("Ads")
plt.ylabel("Number of times each ad was selected")
plt.show()