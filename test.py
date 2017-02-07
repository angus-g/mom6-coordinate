from remapping import mom_remapping
import numpy as np

cs = mom_remapping.Remapping_Cs()
# PCM
cs.remapping_scheme = 0 # PCM

# PPM-H4
cs.remapping_scheme = 2
cs.degree = 2

h0 = 0.75 * np.ones(4)
h1 = 1 * np.ones(3)
h2 = 0.5 * np.ones(6)

u0 = np.array([9., 3., -3., -9.])

print('expected:', [8., 0., -8.])
print(mom_remapping.remapping_core_h(h0, u0, h1, cs))

print('expected:', [10., 6., 2., -2., -6., -10.])
print(mom_remapping.remapping_core_h(h0, u0, h2, cs))
