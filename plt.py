import matplotlib.pyplot as plt
import numpy


plt.figure(figsize=(10, 10))

ticks = [0.1, 1, 4, 16, 32]
x1 = [1, 2, 4, 8, 16]
y1 = [80, 90, 90, 90, 90]

x = numpy.array(x1)
y = numpy.array(y1)

plt.plot(x, y, 'black', lw=2.5)
# plt.plot(x, natural_test_natural, 'black',linestyle='--', lw=2.5)
plt.title('111', fontsize=22)
plt.yticks(size=15)

plt.xlabel('11', fontsize=18)
plt.ylabel('111', fontsize=18)
plt.legend(['11',], fontsize=18)

plt.savefig("./picture/lunwen.png")
plt.show()