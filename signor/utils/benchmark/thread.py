# Created at 5/10/21
# Summary:

import timeit
import matplotlib.pyplot as plt
from signor.utils.system import detect_sys

runtimes = []
threads = [1] + [t for t in range(2, 130, 1)]
for t in threads:
    import torch # important: put it here
    torch.set_num_threads(t)
    print(torch.get_num_threads())
    r = timeit.timeit(setup = "import torch; x = torch.randn(1024, 1024); y = torch.randn(1024, 1024)", stmt="torch.mm(x, y)", number=100)
    runtimes.append(r)

plt.plot(threads, runtimes)
plt.xlabel('n_thread')
plt.ylabel('time')
plt.title(f'{detect_sys()}')
plt.show()