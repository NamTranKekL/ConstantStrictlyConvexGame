import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math

x_mean = []
x_var = []
# Import data from multiple Excel files
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

x_mean = []
x_var = []
x_bound = []
# Loop through file names data2.xlsx to data9.xlsx
for n in range(2, 11):
    excel_file = "data" + str(n) + ".xlsx"  # Update with your Excel file name/path
    df = pd.read_excel(excel_file)

    x_mean.append(df.Epochs.mean())
    x_var.append(math.sqrt(np.var(df.Epochs)))    #plt.show()
    x_bound.append(5000*(n**3))


xES_mean = []
xES_var = []
xES_bound = []

for n in range(2, 11):
    excel_file = "dataSC" + str(n) + ".xlsx"  # Update with your Excel file name/path
    df = pd.read_excel(excel_file)

    xES_mean.append(df.Epochs.mean())
    xES_var.append(math.sqrt(np.var(df.Epochs)))    #plt.show()
    xES_bound.append(5000*(n**3))

cn0 = 1.5
bn0 = (np.log(x_mean[0]*4) - cn0)/2
# plot the log mean
plt.figure(figsize=(6, 6))  # Adjust size if needed
#plt.plot(range(2, 11), np.log(x_mean), color='blue', label='Mean of number of rounds')
plt.plot(range(2, 11), np.log(x_mean*pow(np.arange(2, 11),2)), '--bo', label='Number of rounds')
plt.fill_between(range(2, 11), np.log((x_mean - np.array(x_var))*pow(np.arange(2, 11),2)), np.log((np.array(x_mean) + np.array(x_var))*pow(np.arange(2, 11),2)), color='skyblue', alpha=0.4, label='Variance')
#plt.plot(range(2, 11),  np.log(2*np.arange(2, 11)*(x_mean[0])*np.exp(np.arange(9))), color='green', label='Hypothetical exponential growth curve')
plt.plot(range(2, 11),  bn0*range(2, 11) + cn0, color='green', label='Hypothetical exponential growth curve')

#plt.plot(range(2, 11),  np.log(pow(np.arange(2, 11),5)*(x_mean[0])/8), color='red', label='Upper Bound Growth Curve')

#plt.plot(range(2, 11),  np.log((pow(np.arange(2, 11),5)*(x_mean[0])/8)/x_mean*pow(np.arange(2, 11),2), color='red', label='Upper Bound Growth Curve')

#plt.plot(range(2, 11), np.log(xES_mean*pow(np.arange(2, 11),2)), '--bo', label='Number of rounds ES')
#plt.plot(range(2, 11),  np.log(pow(np.arange(2, 11),5)*(xES_mean[0])/8), color='red', label='Upper Bound Growth Curve ES')


#plt.yscale('log')
#plt.plot(range(2, 11), np.log(x_bound), color='red', label='theoretical bound')

plt.xlabel('$n$ - number of players')  # Add label for x-axis
plt.ylabel('$\ln(T)$')  # Add label for y-axis
#plt.ylabel('epochs')  # Add label for y-axis
plt.title('Strict convexity constant is $0$')  # Add title
plt.legend()  # Show legend
plt.grid(True)  # Add grid
#plt.yscale('log')
plt.savefig('log(T).png', dpi=300)

#plt.plot(range(2, 11),  (pow(np.arange(2, 11),5)*(x_mean[0])/8)/(x_mean*pow(np.arange(2, 11),2)), color='black', label='Upper Bound Growth Curve')

#plt.show()

#plt.plot(range(2, 11),  (pow(np.arange(2, 11),3)*(x_mean[0])/8)/(x_mean*pow(np.arange(2, 11),2)), color='black', label='Upper Bound Growth Curve')
#plt.show()


plt.figure(figsize=(6, 6))  # Adjust size if needed
#plt.plot(range(2, 11), np.log(x_mean), color='blue', label='Mean of number of rounds')
plt.plot(range(2, 11), x_mean*pow(np.arange(2, 11),2), '--bo', label='Number of rounds')
plt.fill_between(range(2, 11), (x_mean - np.array(x_var))*pow(np.arange(2, 11),2), (np.array(x_mean) + np.array(x_var))*pow(np.arange(2, 11),2), color='skyblue', alpha=0.4, label='Variance')
#plt.plot(range(2, 11),  2*np.arange(2, 11)*(x_mean[0])*np.exp(np.arange(9)), color='green', label='Hypothetical exponential growth curve')
plt.plot(range(2, 11),  np.exp(bn0*range(2, 11) + cn0), color='green', label='Hypothetical exponential growth curve')

#plt.plot(range(2, 11),  pow(np.arange(2, 11),5)*(x_mean[0])/8, color='red', label='Upper Bound Growth Curve')

plt.xlabel('$n$ - number of players')  # Add label for x-axis
plt.ylabel('$T$')  # Add label for y-axis
#plt.ylabel('epochs')  # Add label for y-axis
plt.title('Number of samples')  # Add title
plt.legend()  # Show legend
plt.grid(True)  # Add grid
plt.yscale('log')
plt.savefig('T.png', dpi=300)

#plt.plot(range(2, 11),  (pow(np.arange(2, 11),5)*(x_mean[0])/8)/(x_mean*pow(np.arange(2, 11),2)), color='black', label='Upper Bound Growth Curve')

plt.show()