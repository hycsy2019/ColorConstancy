import matplotlib.pyplot as plt

x=[]
y=[]
with open('loss.txt','r') as f:
    s=f.readline()
    while s:
        s=s.split(' ')
        x.append(int(s[0]))
        y.append(float(s[1]))
        s=f.readline()
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.plot(x,y)
plt.show()