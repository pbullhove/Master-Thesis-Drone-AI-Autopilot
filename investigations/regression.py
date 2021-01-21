import numpy as np
from sklearn.linear_model import LinearRegression

h = np.array([1.0, 3.0, 5.0, 7.0]).reshape((-1,1))

l = np.array([0.1, 1.0, 1.9, 2.8])
w = np.array([0.7, 2.8, 4.9, 7.0])

model_l = LinearRegression()
model_w = LinearRegression()

# print(h)
# print(l)
# print(w)

model_l.fit(h, l)
model_w.fit(h, w)

r_sq_l = model_l.score(h, l)
r_sq_w = model_w.score(h, w)

# print('coefficient of determination l:', r_sq_l)
# print('coefficient of determination w:', r_sq_w)

# print('intercept l:', model_l.intercept_)
# print('slope l:', model_l.coef_)

# print('intercept w:', model_w.intercept_)
# print('slope w:', model_w.coef_)




"""
x + 1.0
y + 0
z + 0.54

###############

y: 4.79761961546
z: 8.01943131059

y: 0.653771108556
z: 5.85646606726    

y: 0.458669385835
z: 4.86567657934

y: 0.183204234685
z: 3.45168121425

y: 0.38501080007
z: 2.34045488661

y: 0.593000101001
z: 1.58433231618

###############

x: -2.00270098264
z: 8.10801583763


###############

[[-0.6076217294918058, 1.3984431181983699, 2.2222004870919476], [-0.3331547862232188, 0.7040497440582005, 1.3169372912482706], [-1.0571417824447191, 2.034951566897678, 3.1362479596794333], [-1.983537056296233, 3.4486734631514864, 4.948833735812898], [-3.0704142661157943, 5.472726587928907, 7.4157437497964915]]
"""

def do_regression():
    data = np.array([[0.6076217294918058, 1.3984431181983699, 2.2222004870919476],
                    [0.3331547862232188, 0.7040497440582005, 1.3169372912482706],
                    [1.0571417824447191, 2.034951566897678, 3.1362479596794333],
                    [1.983537056296233, 3.4486734631514864, 4.948833735812898],
                    [3.0704142661157943, 5.472726587928907, 7.4157437497964915]])

    h = data[:,2].reshape((-1,1))

    l = data[:,0]
    w = data[:,1]
    
    # h = np.array([1.0, 3.0, 5.0, 7.0]).reshape((-1,1))

    # l = np.array([0.1, 1.0, 1.9, 2.8])
    # w = np.array([0.7, 2.8, 4.9, 7.0])

    model_l = LinearRegression()
    model_w = LinearRegression()

    print(h)
    print(l)
    print(w)

    model_l.fit(h, l)
    model_w.fit(h, w)

    r_sq_l = model_l.score(h, l)
    r_sq_w = model_w.score(h, w)

    print('coefficient of determination l:', r_sq_l)
    print('coefficient of determination w:', r_sq_w)

    print('intercept l:', model_l.intercept_)
    print('slope l:', model_l.coef_)

    print('intercept w:', model_w.intercept_)
    print('slope w:', model_w.coef_)

    """
    ('intercept l:',    -0.3464550733967431
    ('slope l:',        0.46135304
    ('intercept w:',    -0.36154347233387263
    ('slope w:',        0.78080833

    """


do_regression()