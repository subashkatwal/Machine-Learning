
import matplotlib.pyplot as plt 
import numpy as np
'''
year=[2010,2012,2014,2016,2018,2020,2022,2024]
gpa=[3.9,3.9,3.8,3.9,3.70,3.40,3.26,3.50]

plt.plot(year,gpa,color='green',marker='o',linestyle='solid')
plt.title("GPA Overview")
plt.xlabel('X-axis')
plt.ylabel('X-axis')
plt.show()
'''



#simple test data 
'''
x=[1,2,3]
y=[3,4,5]
z=x+y

plt.plot(x,y,marker='o',linestyle="solid", color="red")
plt.plot(z,marker='o',linestyle="solid")
plt.xlabel("X-axis")
plt.ylabel("X-axis")

plt.show()

'''




'''Bar Graph in python'''
# x=[2,3,4,5,6,7,8]
# y=[10,11,12,13,14,15,16]
# plt.bar(x,y)
# plt.legend("Bar")

# plt.show()


'''Scatter plot in python '''
'''
x=[2,4,5,6,7,8,9,0,12]
y=[23,24,23,21,11,17,87,1,2]
plt.scatter(x,y,marker='o',linestyle="solid",color='red')
plt.grid(True)
plt.show()

'''


'''Stack plot '''
'''
days=[1,2,3,4,5]
studying=[7,8,2,1,3]
playing=[4,5,3,6,1]

plt.stackplot(days,studying,playing,colors=['r','c'])
plt.xlabel("Days")
plt.ylabel("No of hours")

plt.title("Histogram of activities")
plt.show()

'''

"""
np.random.seed(10)
data=np.random.normal(10,20,200)
fig=plt.figure(figsize=(10,7))
plt.boxplot(data)
plt.show()
"""


'''Pie chart '''
'''
no_of_employee=[10,30,20,36,9]
fields=['Frontend Developer','Backend Developer','Network Engineer','AI/Ml Engineer','Data Analytics']
e=(0.1,0,0,0,0)
plt.figure(figsize=(10, 8))
plt.pie(no_of_employee,explode=e,labels=fields,autopct=lambda p: f'{p:.1f}%',colors=['r','c','g','pink','blue'])
plt.title("Employee Analysis")
plt.legend(fields,loc='upper left')
plt.show()

'''


"""Error plot """
x=[1,2,3,4,5,6]
y=[6,7,8,9,10,11]
yError=0.2
plt.plot(x,y)
plt.errorbar(x,y,yerr=yError,fmt='o')
plt.show()
