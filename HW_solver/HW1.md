# HW1
## Problem 1
as the problem descirption tells,we can abstract the problem as the question:

P(t+1)=αMP(t)+(1-α)v,P is a 5-dimensional column vector,v is a fixed 5-dimensional column vector,M is a known n*n matrix, Alpha is a constant, if the P1: n = [1/3, 1/3, 1, 0, 3], the P2: v = [0, 1/3, 1/3, 1, 3], P3:1/3,0,0,1/3, 1/3, P4: v =,0,0,0,0,0 [1], when this equation iteration convergence hypothesis has been l P1, P2, P3, P4, so when v respectively under known P1-4 =,1,0,0,0 [0], [0,0,0,0,1], [0.1, 0.2, 0.3, 0.2, 0.2], P can use known vector said after convergence, how should if you can say

So we can fomulate the problem, as an equation:
    P = ⍺MP + (1-⍺)v
With the linear algebra's technique,
        we can slove the equation and simplify it as:
                $P = (I-⍺M)^(-1)(1-⍺)v$
Thus, the P is linked with the column vector v,
            1/3a + 0    + 1/3c + d  =0
            1/3a + 0    + 0    + 0  =1
            1/3a + 1/3b + 0    + 0  =0
            0    + 1/3b +1/3c  + 0  =0
            0    + 1/3b +1/3c  + 0  =0 
    so,
        the P(E)= 3P1-3P2+3P3-2P4
however,Felicity's P cannot be solved, as its det is 0,a singular matrix.
    For Glynnis,in a similar way,
        we can list as:
            1/3a + 0    + 1/3c + d  =0.1
            1/3a + 0    + 0    + 0  =0.2
            1/3a + 1/3b + 0    + 0  =0.3
            0    + 1/3b +1/3c  + 0  =0.2
            0    + 1/3b +1/3c  + 0  =0.2
    so,
        P(G) = 0.6P1 + 0.3P2 + 0.3P3 - 0.2P4 
## Problem 2
Prof:
    Since:
        $r = Ar$
    Consider:
       $A = βM + \frac{1-β}{N}II^T$
    Thus,
        $r = (βM + \frac{1-β}{N}II^T)r$
    And we can simplify the equation with linear algebra,
        Since: 
            $I^{T}r =  \sum_{i=1}^{N}r(i)$
        Consider:r is a normalized column vector,
                    $\sum_{i=1}^{N}r(i)=1$
    Hence,  
        $r = {\beta}Mr+\frac{1-β}{N}I$
        $Q.E.D$

## Problem 3
    ==3.1:== P3 = .5939670138
    ==3.2:== P4 = .5209960937
    ==3.3:== P8 = .3815917968

    ==3.4:== 2(.544921875),3(.5939670138),4(.5209960937),
    5(.5112711588),6(.6140724464),7(.5209960937),10(.6140724464)

## Problem 4
4.1
    Iter1 Initial_LB  LinkV    New_LB
    Node1:0           (0,1)     0
    Node2:1           (0,1)     1
    Node3:1           (0,1)     1
    Node4:1           (0,1)     1
    Node5:1           (1,0)     0
    Node6:0           (1,1)     0 
    Node7:0           (1,1)     0

    Iter2 Initial_LB  LinkV    New_LB
    Node1:0           (0,1)     0
    Node2:1           (0,1)     1
    Node3:1           (1,0)     0
    Node4:1           (0,1)     1
    Node5:0           (1,0)     0
    Node6:0           (1,0)     0 
    Node7:0           (1,1)     0

4.2
    Yes,the graph has been converged

