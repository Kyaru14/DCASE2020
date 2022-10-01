def is2power(n:int) -> bool:
    return (n & n-1) == 0
def to2power(n:int) -> int:
    c = 0
    while n!=1:
        n = n >> 1
        c = c + 1
    for _ in range(c + 1):
        n = n << 1
    return n

def FFT(x):
    n = len(x)
    if not is2power(n):
        n = to2power(n)
        x_new = np.zeros(shape=n,dtype=int)
        x_new[0:len(x)] = x
        x = x_new
    if n == 1:
        return x
    w = np.exp(2*np.pi*1j/n)
    Pe,Po = x[::2],x[1::2]
    Ye,Yo = FFT(Pe),FFT(Po)
    y = [0] * n
    for j in range(int(n/2)):
        y[j] = Ye[j] + w ** j * Yo[j]
        y[j + int(n/2)] = Ye[j] - w ** j * Yo[j]
    return y