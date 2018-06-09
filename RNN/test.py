import  numpy as np
import matplotlib.pyplot as plt
import io

# e^(log(a)) = a
# 当temperature = 1 时，输出原来的概率分布
# 当temperature < 1 时，输出较保守，相当对概率分布做了乘方，概率分布更陡，原来概率大的会更大，小的会更小。
# 当temperature > 1 时，输出较开放，相当对概率分布做了开方，概率分布更平缓，原来概率大会变小，小的会变大，趁于平缓，这样会增加更多的可能性。
# def sample(preds,temperature=1.0):
#     preds = np.asarray(preds).astype("float64")
#     print(preds)
#     preds = np.log(preds)/temperature
#     print(preds)
#     exp_preds = np.exp(preds)
#     print(exp_preds)
#     preds = exp_preds / np.sum(exp_preds)
#     print(preds)
#     return preds
#
# preds = [0.8,0.15,0.05]
#
# probas_1 = sample(preds,temperature=1)
# probas_2 = sample(preds,temperature=0.5)
# probas_3 = sample(preds,temperature=1.5)
#
# plt.plot(range(len(probas_1)),probas_1,color="red")
# plt.plot(range(len(probas_2)),probas_2,color="green")
# plt.plot(range(len(probas_3)),probas_3,color="blue")
#
# plt.show()

Tx = 40

usr_input = input("Write the beginning of your poem, the Shakespeare machine will complete it. Your input is: ")
# zero pad the sentence to Tx characters.
sentence = ('{0:0>' + str(Tx) + '}').format(usr_input).lower()

print(sentence)