import numpy as np
import matplotlib.pyplot as plt

def main():
    filepath = "../processed/correct_within_1_try2.npy"
    Y = np.load(filepath)[:,1:]
    Y = np.absolute(Y - 1)
    print(np.shape(Y))
    
    M = np.mean(Y, axis=0)
    print(np.shape(M))
    
    plt.plot(M)
    plt.xlabel("Timestep")
    plt.ylabel("Fraction of failure labels")
    plt.savefig("data_balance.pdf")


if __name__=="__main__":
    main()
