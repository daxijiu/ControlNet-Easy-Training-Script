from tutorial_dataset import MyDataset

dataset = MyDataset()
print('训练集载入完成，本次训练集有' + str(len(dataset)) + '个样本')

item = dataset[1]
jpg = item['jpg']
txt = item['txt']
hint = item['hint']

#print(txt)
#print(jpg.shape)
#print(hint.shape)
