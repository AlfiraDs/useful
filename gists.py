

# row idx and col idx from range(n)
n = 5
n_cols = 2
for i in range(n):
    print('row:', i // n_cols)
    print('col:', i % n_cols)
