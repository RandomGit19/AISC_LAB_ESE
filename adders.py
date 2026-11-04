
class MPNeuron:

    def __init__(self,weights,threshold):
        self.weights = weights
        self.threshold = threshold

    def predict(self,inputs):
        summation = sum(x*w for x,w in zip(inputs,self.weights))

        return 1 if summation >=self.threshold else 0 

    


#AND GATE

AND = MPNeuron([1,1],2)


#OR GATE

OR = MPNeuron([1,1],1)


#NAND GATE

NAND =MPNeuron([-1,-1],-1)

#XOR GATE 

def xor(a,b):

    p = NAND.predict([a,b])
    q = OR.predict([a,b])

    return AND.predict([p,q])


def half_adder(a,b):

    sum = xor(a,b)
    carry = AND.predict([a,b])

    return sum,carry

def full_adder(a,b,cin):

    s1,c1 = half_adder(a,b)

    s2,c2 = half_adder(s1,cin)

    carry = OR.predict([c1,c2])

    return s2,carry 


def main():

    print("Half Adder ")

    print("A B | Sum Carry")

    for a in [0,1]:
        for b in [0,1]:

            s,c = half_adder(a,b)

            print(f"{a} {b} | {s} {c}")

    print("Full Adder")

    print("A B Cin | Sum Carry")

    for a in [0,1]:
        for b in [0,1]:
            for cin in [0,1]:

                s,c = full_adder(a,b,cin)
                print(f"{a} {b} {cin} | {s} {c}")

    print("Thanks")

main()



