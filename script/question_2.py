#Paolo Tamagnini
#paolotamag@gmail.com
#1536242 - Data Science

import numpy as np
import pandas as pd
import time
import scipy.optimize
import random
import matplotlib.pyplot as plt

def obj_f(x):
    return (0.5 * np.dot(x.T, np.dot(Q, x))- np.dot(c, x))

def poly_kernel(xi,xj):
    arg = np.dot(xi,xj)+1
    return arg**p

def grad(lambs):
    return np.dot(Q,lambs) - c

def func_m(R_KKT_values):
    if len(R_KKT_values) != 0:
        return max(R_KKT_values)
    else:
        return -np.inf
    
def func_M(S_KKT_values):
    if len(S_KKT_values) != 0:
        return min(S_KKT_values)
    else:
        return +np.inf

def loss_mini(x, sign=1.):
    return sign * (0.5 * np.dot(x.T, np.dot(Q_mini, x))- np.dot(c_mini, x))

def jac_mini(x, sign=1.):
    return sign * (np.dot(x.T, Q_mini) - c_mini)


def big_deals(lambda_star,x_use_it):
    big_deal = 0
    for i in range(0,len_train):
        big_deal+=lambda_star[i]*y_train[i]*poly_kernel(x_use_it,x_train[i])
    return big_deal

def find_b(lambda_star,l=0):
    looking = True
    while looking:
        x_to_use = x_train[l]
        y_to_use = y_train[l]
        if lambda_star[l] >0 and lambda_star[l] < C:
            looking = False
        else:
            l+=1
    b_star = 1/y_to_use - big_deals(lambda_star,x_to_use)
    return b_star



def compute_pred(lambdax,x_new,b_star):

    sgn = big_deals(lambdax,x_new)+b_star
    y_pred=[]
    if sgn >= 0:
        y_pred = 1.0
    elif sgn < 0:
        y_pred = -1.0
    return y_pred

def  compute_error(xi,yi,lambdas):
    be =  find_b(lambdas,0)
    len_vect = len(yi)
    somma = 0
    y_pred_vector = []
    for i in range(0,len_vect):
        y_hat = compute_pred(lambdas,xi[i],be)
        y_pred_vector.append(y_hat)
        somma += np.abs(y_hat - yi[i])/2
    return somma / float(len_vect)



pixels = range(0,256)
df_0 = pd.read_csv('../digit-data/train.0',header=None,sep=',', names = pixels)
num_0 = len(df_0)
df_8 = pd.read_csv('../digit-data/train.8',header=None,sep=',', names = pixels)
num_8 = len(df_8)
df_1 = pd.read_csv('../digit-data/train.1',header=None,sep=',', names = pixels)
num_1 = len(df_1)

print '--- DIGIT DATA ---'
print
print '# of 0s: ', num_0
print '# of 1s: ', num_1
print '# of 8s: ', num_8
print
print '-----------------'

df_0_GrT = df_0
df_0_GrT['G_T'] = np.repeat(True,num_0)
df_0_GrT['G_T_val'] = np.repeat(1.0,num_0)

df_1_GrT = df_1
df_1_GrT['G_T'] = np.repeat(False,num_1)
df_1_GrT['G_T_val'] = np.repeat(-1.0,num_1)

df_8_GrT = df_8
df_8_GrT['G_T'] = np.repeat(False,num_8)
df_8_GrT['G_T_val'] = np.repeat(-1.0,num_8)

df_0_1 = df_0_GrT.append(df_1_GrT)
df_0_1 = df_0_1.sample(frac=1).reset_index(drop=True)
num_0_1 = num_0 + num_1

df_0_8 = df_0_GrT.append(df_8_GrT)
df_0_8 = df_0_8.sample(frac=1,random_state=12345).reset_index(drop=True)
num_0_8 = num_0 + num_8

percTrainTest = 0.7
df_train = df_0_8[:int(num_0_8*percTrainTest)].reset_index(drop=True)
df_test = df_0_8[int(num_0_8*percTrainTest):].reset_index(drop=True)

#df_train = df_0_1[:int(num_0_1*percTrainTest)].reset_index(drop=True)
#df_test = df_0_1[int(num_0_1*percTrainTest):].reset_index(drop=True)

len_train = len(df_train)
len_test = len(df_test)
y_train = np.array(df_train['G_T_val'])
x_train = np.array(df_train[pixels])
y_test = np.array(df_test['G_T_val'])
x_test = np.array(df_test[pixels])

C=10
p=2

print 'current C:',C
print 'current p:',p
print
Y_X_poly = np.zeros((len_train,len_train))
for i in range(0,len_train):
    for j in range(0,len_train):
        y_prod = y_train[i]*y_train[j]
        Y_X_poly[i,j] += y_prod*poly_kernel(x_train[i],x_train[j])

print 'Matrix Q computed..'
print
# current kernel = polynomial



Q = Y_X_poly
c = np.repeat(+1,len_train)
A = np.concatenate((-1*np.identity(len_train),np.identity(len_train)))
b = np.concatenate((np.zeros(len_train),np.repeat(C,len_train)))
Z = y_train

cons = ({'type':'ineq',
        'fun':lambda x: b - np.dot(A,x),
        'jac':lambda x: -A},
       {'type':'eq',
        'fun':lambda x: np.dot(Z,x),
        'jac':lambda x: Z})

iteraz = 0
tooManyIter = 100
eps = 1.94
random.seed(12345)
train_index = np.arange(len_train)
lambda_now = np.zeros(len_train)

m_and_Ms = []
ms = []
Ms = []
funs = []
allErrorsTrain = []
allErrorsTest = []

print "It will require more time to compute errors and such things for each iteration."
print "Anyway those values are required to plot the progress of the algorithm."

while True:

	lol = raw_input("Do you want to plot despite the increase of waiting time? (Y/N) >> ")

	if lol in ['Y', "N"]:
		break

if lol in ['Y']:
	plotLOL = True

elif lol in ['N']:
	plotLOL = False

st = time.time()

while True:
    
    if iteraz == tooManyIter:
        print "too many iter"
        break

    grad_now = grad(lambda_now)
    KKT = -grad_now/y_train
    L = set([i for i in train_index if lambda_now[i] == 0])
    U = set([i for i in train_index if lambda_now[i] == C])
    limbo = set([i for i in train_index if lambda_now[i] < C and lambda_now[i] > 0])
    Lplus = set([i for i in L if y_train[i] > 0])
    Lminus = set([i for i in L if y_train[i] < 0])
    Uplus = set([i for i in U if y_train[i] > 0])
    Uminus = set([i for i in U if y_train[i] < 0])
    R = Lplus | Uminus | limbo
    R_KKT_values = KKT[list(R)]
    S = Lminus| Uplus | limbo
    S_KKT_values = KKT[list(S)]

    I = set([i for i in R if KKT[i] == max(R_KKT_values)])
    J = set([i for i in S if KKT[i] == min(S_KKT_values)])

    m = func_m(R_KKT_values)
    M = func_M(S_KKT_values)
    m_and_Ms.append(m-M)
    ms.append(m)
    Ms.append(M)
    if m <= M + eps:
        print "optimality reached"
        break
    
    i = random.sample(I,1)[0]
    j = random.sample(J,1)[0]
    W = set([i,j])
    
    q = 2
    Q_mini = np.array([[Q[i,i],Q[i,j]],[Q[j,i],Q[j,j]]])
    c_mini = np.ones(q)
    Z_mini = y_train[[i,j]]
    A_mini = np.concatenate((-1*np.identity(q),np.identity(q)))
    b_mini = np.concatenate((np.zeros(q),np.repeat(C,q)))
    cons_mini = ({'type':'ineq',
            'fun':lambda x: b_mini - np.dot(A_mini,x),
            'jac':lambda x: -A_mini},
           {'type':'eq',
            'fun':lambda x: np.dot(Z_mini,x),
            'jac':lambda x: Z_mini})
    x0 = np.repeat(0.0,q)

    res_SLSQP_mini = scipy.optimize.minimize(loss_mini, x0, jac=jac_mini,constraints=cons_mini, method='SLSQP')
    lambda_now[i] = res_SLSQP_mini['x'][0]
    lambda_now[j] = res_SLSQP_mini['x'][1]



    if plotLOL:
	    funs.append(obj_f(lambda_now))
	    error_test = compute_error(x_test,y_test,lambda_now)
	    allErrorsTest.append(error_test)
	    error_train =compute_error(x_train,y_train,lambda_now)
	    allErrorsTrain.append(error_train)
    if iteraz%10 == 0:
        print "# iter: ", iteraz
    iteraz +=1


if plotLOL:
	plt.plot(allErrorsTest,marker="o",label="test error")
	plt.plot(allErrorsTrain,marker="o",label="train error")
	plt.title("Errors for each SVM_light cycle",fontsize = 25)
	plt.xlabel('iterations',fontsize = 20)
	plt.ylabel('errors',fontsize = 20)
	plt.legend(fontsize = 20)
	plt.show()

	plt.plot(funs,marker="o",label="obj. funct.")
	plt.title("funct. eval. for each SVM_light cycle",fontsize = 25)
	plt.xlabel('iterations',fontsize = 20)
	plt.ylabel('obj. funct.',fontsize = 20)
	plt.legend(fontsize = 20)
	plt.show()

	plt.plot(m_and_Ms,marker="o",label="m - M")
	plt.title("delta for each SVM_light cycle",fontsize = 25)
	plt.xlabel('iterations',fontsize = 20)
	plt.ylabel('m - M',fontsize = 20)
	plt.legend(fontsize = 20)
	plt.show()
error_test = compute_error(x_test,y_test,lambda_now)
error_train =compute_error(x_train,y_train,lambda_now)
print "elapsed time:", time.time() - st , "s"
print "Error on test set:",error_test
print "Error on train set:",error_train
print "# iterations:",iteraz
print "# of gradient evaluations:", iteraz
print "objective function minimum value:",obj_f(lambda_now)
print 

