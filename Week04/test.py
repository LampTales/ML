import matplotlib.pyplot as plt

x = list(range(1, 1001))
y_1 = [max(20000*i/15, 20000/4) for i in x]
y_2 = [max(20000/15, 20000/4, i*20000/(15 + 0.1*i)) for i in x]
y_3 = [max(20000/15, 20000/4, i*20000/(15 + 0.6*i)) for i in x]
y_4 = [max(20000/15, 20000/4, i*20000/(15 + 4*i)) for i in x]

# plot x and y
plt.plot(x, y_1, linewidth=2, label='CS')
plt.plot(x, y_2, linewidth=2, label='P2P 0.1M')
plt.plot(x, y_3, linewidth=2, label='P2P 0.6M')
plt.plot(x, y_4, linewidth=2, label='P2P 4M')
plt.legend()
plt.show()