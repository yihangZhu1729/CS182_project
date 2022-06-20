import matplotlib.pyplot as plt

flex_250 = [0.2405, 0.1956, 0.3935, 0.4676, 0.4919, 0.4964, 0.5073, 0.512, 0.5109, 0.5166, 0.5191, 0.5158, 0.5198, 0.5136, 0.5176, 0.5198, 0.5189, 0.5133, 0.5196, 0.5238, 0.5227, 0.5197, 0.5155, 0.5262, 0.5205, 0.5217, 0.5202, 0.5236, 0.5216, 0.5191, 0.5197, 0.5181, 0.5221, 0.5238, 0.5161, 0.5213, 0.5198, 0.5198, 0.5217, 0.52]
ours_250 = [0.103, 0.312, 0.3897, 0.4841, 0.5013, 0.5094, 0.5122, 0.5229, 0.5263, 0.5321, 0.5371, 0.5392, 0.5494, 0.5464, 0.5473, 0.545, 0.5468, 0.549, 0.5459, 0.5462, 0.5453, 0.5472, 0.5467, 0.5458, 0.5447, 0.5502, 0.5509, 0.5497, 0.5462, 0.5438, 0.5471, 0.5516, 0.5464, 0.5442, 0.5511, 0.5457, 0.547, 0.5423, 0.548, 0.5466]

x = [5*i+5 for i in range(40)]

plt.title("Top-1 acc with 25 labels per class")
plt.xlabel("epoch")
plt.ylabel("ACC")
plt.plot(x, flex_250, marker='o', markersize=3)
plt.plot(x, ours_250, marker='o', markersize=3)

for a, b in zip(x, flex_250):
    if a % 25 == 0:
        plt.text(a, b, b, ha='center', va='bottom', fontsize=10)

for a, b in zip(x, ours_250):
    if a % 25 == 0:
        plt.text(a, b, b, ha='center', va='bottom', fontsize=10)


plt.legend(['FlexMatch', 'Ours', 'ours_stodepth'])
plt.show()