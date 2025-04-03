// function to perform bit reversal permutation
int reverseBits(int n, int numBits)
{
    int result = 0;
    for (int i = 0; i < numBits; i++) {
        if ((n >> i) & 1) {
            result |= 1 << (numBits - 1 - i);
        }
    }
    return result;
}

void fft_radix2(int xn[], complex<double> Xk[], int n, int numBits)
{
    // bit reversal permutation
    for (int i = 0; i < n; i++)
    {
        int reversedIndex = reverseBits(i, numBits);
        if (i < reversedIndex) {
            swap(xn[i], xn[reversedIndex]);
        }
    }

    // Cooley-Tukey algorithm
    for (int len = 2; len <= n; len <<= 1)
    {
        double angle = -2 * M_PI / len;
        complex<double> wlen(cos(angle), sin(angle));
        for (int i = 0; i < n; i += len)
        {
            complex<double> w(1, 0);
            for (int j = 0; j < len / 2; j++)
            {
                complex<double> u(xn[i + j],0);
                complex<double> v(xn[i + j + len / 2],0);
                complex<double> t = w * v;
                Xk[i + j] = u + t;
                Xk[i + j + len / 2] = u - t;
                w *= wlen;
            }
        }
    }
}
