# from sklearn.linear_model import LogisticRegression
# x=[[1, 2],[2,3],[3,4],[4,5],[5,6] ,[6,7]]
# y=[0,1,0,1,0,1]

# #Train model
# model=LogisticRegression()
# model.fit(x,y)

# #make prediction
# prediction=model.predict([[7,8]])[0]
# print(prediction)


# from sklearn.datasets import make_blobs
# from matplotlib import pyplot as plt 
# from matplotlib import style

# style.use("fivethirtyeight")
# x,y=make_blobs(n_samples=100,centers= 3,cluster_std=1,n_features=2)
# plt.scatter(x[:,0],x[:,1],s=40,color='g')
# plt.xlabel("X")
# plt.ylabel("Y")
 
# plt.show()
# plt.clf()

#making moon
# from sklearn.datasets import make_moons
# from matplotlib import pyplot as plt 
# from matplotlib import style

# x,y=make_moons(n_samples= 1000,noise=0.1)
# plt.scatter(x[:,0],x[:,1],s=40,color='b')
# plt.xlabel("x")
# plt.ylabel("y")

# plt.show()
# plt.clf()



#circles
from sklearn.datasets import make_circles
from matplotlib import pyplot as plt 
from matplotlib import style

hex_color = '#FF0000'
x,y=make_circles(n_samples=1000,noise=0.002)
plt.scatter(x[:,0],x[:,1],s=40,color=hex_color)
plt.xlabel("x")
plt.ylabel("y")

plt.show()
plt.clf()


