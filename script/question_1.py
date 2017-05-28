#Paolo Tamagnini
#paolotamag@gmail.com
#1536242 - Data Science

import numpy as np
import pandas as pd
import time
import scipy.optimize
import matplotlib.pyplot as plt


def gauss_kernel(xi,xj):
    expon = -gamma*np.linalg.norm(xi-xj)**2
    return np.exp(expon)

def poly_kernel(xi,xj):
    arg = np.dot(xi,xj)+1
    return arg**p

###################################################

def loss(x, sign=1.):
    return sign * (0.5 * np.dot(x.T, np.dot(Q, x))+ np.dot(c, x))

def jac(x, sign=1.):
    return sign * (np.dot(x.T, Q) + c)

###################################################

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

bestErrorTrain = 3213123
bestErrorTest = 3213123
bestTime = 234567890
bestRespConst = True
opt_info_best = 'NaN'
bestp = 0
bestC = 0 
allErrorsTrain = []
allErrorsTest = []
times = []
while True:
	lol = raw_input("Do you want to iterate many different combinations of C and p ? (Y/N) >> ")
	if lol in ['Y', "N"]:
		break
if lol in ['Y']:
	p_list = np.array(range(2,10))
	C_list = np.array([0.01,0.1,1.0,10])
	#desired C index to plot
	gh = 3
elif lol in ['N']:
    print "p = 2 --> slow optimization (10 m) but test error = 0.9 %"
    print "p = 3 --> fast optimization (10 s) but test error = 2 %"
    while True:
        lol2 = raw_input("Which p ? (2 or 3) >> ")
        if lol2 in ['2', "3"]:
            break
    if lol2 in ['2']:
    	p_list = np.array([2])
    	C_list = np.array([10])
    	gh = 0
    elif lol2 in ['3']:
        p_list = np.array([3])
        C_list = np.array([10])
        gh = 0 

for C in C_list:
    for p in p_list:
        print '-----------------------'
        print
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
        c = np.repeat(-1,len_train)
        A = np.concatenate((-1*np.identity(len_train),np.identity(len_train)))
        b = np.concatenate((np.zeros(len_train),np.repeat(C,len_train)))
        Z = y_train

        cons = ({'type':'ineq',
                'fun':lambda x: b - np.dot(A,x),
                'jac':lambda x: -A},
               {'type':'eq',
                'fun':lambda x: np.dot(Z,x),
                'jac':lambda x: Z})

        st = time.time()
        print "Computing optimization.."
        x0 = np.repeat(0.0,len_train)

        res_SLSQP = scipy.optimize.minimize(loss, x0, jac=jac,constraints=cons, method='SLSQP')
        timino =  time.time()-st
        print 'Elapsed time: ',timino, 's'
        times.append(timino)
        print res_SLSQP
        lambda_super_star = res_SLSQP['x']
        error_test = compute_error(x_test,y_test,lambda_super_star)
        allErrorsTest.append(error_test)
        error_train =compute_error(x_train,y_train,lambda_super_star)
        allErrorsTrain.append(error_train)
        print
        print 'Error on test:',error_test
        print 'Error on train:',error_train
        print
        constr = True
        for i in range(0,len_train):
            if lambda_super_star[i] <= 0:
                print 'outbound lambda (<0), row #',i
                constr = False
            elif lambda_super_star[i] >= C:
                print 'outbound lambda (>C), row #',i
                constr = False
        if np.dot(lambda_super_star,y_train) != 0:
            print 
            constr = False
        print 
        print
        if error_test < bestErrorTest:
            bestErrorTrain = error_train
            bestErrorTest = error_test
            bestRespConst = constr
            opt_info_best = res_SLSQP
            bestp = p
            bestC = C
            bestTime = timino
print
print '-----------------------'
print '-----------------------'
print '-----------------------'
print
print ' The best error is achieved for (C,p): ',(bestC,bestp)
if bestRespConst:
    print "the constraint are of course respected"
else: 
    print "but the constraint are not respected"
print
print "Elapsed time:",bestTime,"s"
print 'Error on test:',bestErrorTest
print 'Error on train:',bestErrorTrain
print
print opt_info_best


error_test_matrix =np.array(allErrorsTest).reshape(len(C_list),len(p_list))
error_train_matrix =np.array(allErrorsTrain).reshape(len(C_list),len(p_list))
dfIterTest = pd.DataFrame(error_test_matrix)
dfIterTrain = pd.DataFrame(error_train_matrix)
dfIterTest.columns = ["p: "+str(l) for l in p_list]  
dfIterTest.index = ["C: "+str(l) for l in C_list] 
dfIterTrain.columns = ["p: "+str(l) for l in p_list]  
dfIterTrain.index = ["C: "+str(l) for l in C_list]
print 
print "Test errors:"
print dfIterTest
print
print "Train errors:"
print dfIterTrain

plt.plot(p_list, error_test_matrix[gh,:],marker="o",label="test error")
plt.plot(p_list, error_train_matrix[gh,:],marker="o",label="train error")
plt.title("Errors with C = 10",fontsize = 25)
plt.xlabel('p',fontsize = 20)
plt.ylabel('errors',fontsize = 20)
plt.legend(fontsize = 20)
plt.show()