import numpy as np
import gsw

p_ref = 2000
compr = 0.01
max_iter = 8
max_tol = 1e-12
eps = 1e-6

def hycom(h, sa, ct, targ_dens, dz_75, max_int_depth, max_lay_thick,
          s_rule=False, s_rule_alt=False, s_topo=False, detangle=False):
    """
    Build a HyCOM1 grid from thicknesses h, absolute salinity sa
    and conservative temperature ct.
    """

    # calculate interface positions, summing downward
    z = np.concatenate((np.zeros((1, h.shape[1])), h.cumsum(axis=0)), axis=0)
    # allocate the output positions
    # copy here so we have the correct surface and bottom
    z_new = z.copy()

    # calculate pressure
    p = p_ref + compr * ((z[1:,:] + z[:-1,:]) / 2 - p_ref)

    # calculate density
    r = gsw.rho(sa, ct, p)

    # enforce monotonicity, preserving the deepest density
    for k in range(r.shape[0]-1, 0, -1):
        # update all values above the bottom
        r[k-1,:] = np.minimum(r[k-1,:], r[k,:])

    # calculate interpolation edge values and coefficients
    e = np.empty(h.shape + (2,))
    for k in range(1, h.shape[0]):
        # loop skips leftmost cell

        # cell thicknesses
        h0 = h[k-1,:]
        h1 = h[k,:]

        # set vanished layers to cut-off minimum thickness
        m = h0 + h1 == 0
        h0[m] = 1e-10
        h1[m] = 1e-10

        # left edge of current cell
        e[k,:,0]  = (r[k-1,:]*h1 + r[k,:]*h0) / (h0 + h1)
        # right edge of previous cell
        e[k-1,:,1] = e[k,:,0]

    # boundaries are simply boundary cell averages
    e[0,:,0]  = r[0,:]
    e[-1,:,1] = r[-1,:]

    # bound edge values with limiter
    for k in range(e.shape[0]):
        # handle boundaries
        if k == 0:
            k0 = k
            k1 = k
            k2 = k + 1
        elif k == e.shape[0] - 1:
            k0 = k - 1
            k1 = k
            k2 = k
        else:
            k0 = k - 1
            k1 = k
            k2 = k + 1

        # thicknesses
        h_l = h[k0,:]
        h_c = h[k1,:]
        h_r = h[k2,:]

        # value
        u_l = r[k0,:]
        u_c = r[k1,:]
        u_r = r[k2,:]

        # edges (before bounding)
        u0_l = e[k,:,0]
        u0_r = e[k,:,1]

        # slopes
        s_l = 2 * (u_c - u_l) / (h_c + 1e-30)
        s_c = 2 * (u_r - u_l) / (h_l + 2*h_c + h_r + 1e-30)
        s_r = 2 * (u_r - u_c) / (h_c + 1e-30)

        # NB: this is converted to work on all columns simultaneously
        slope = np.sign(s_c) * np.minimum(np.abs(s_l),
                                          np.abs(s_c),
                                          np.abs(s_r))
        # no slope at local extremum
        slope[s_l * s_r <= 0] = 0

        # convert to local coordinate system
        slope *= h_c / 2

        # left and right limits
        lim_l = u_c - np.sign(slope) * np.minimum(np.abs(slope),
                                                  np.abs(u0_l - u_c))
        lim_r = u_c + np.sign(slope) * np.minimum(np.abs(slope),
                                                  np.abs(u0_r - u_c))
        # apply limits
        np.putmask(u0_l, (u_l - u0_l) * (u0_l - u_c) < 0, lim_l)
        np.putmask(u0_r, (u_r - u0_r) * (u0_r - u_c) < 0, lim_r)

        # bound by neighbouring cell means
        u0_l = np.maximum(np.minimum(u0_l, np.maximum(u_l, u_c)),
                          np.minimum(u_l, u_c))
        u0_r = np.maximum(np.minimum(u0_r, np.maximum(u_r, u_c)),
                          np.minimum(u_r, u_c))

        # save updated edge values
        e[k,:,0] = u0_l
        e[k,:,1] = u0_r

    # average discontinuous edge values
    # loop over interior edges
    for k in range(e.shape[0] - 1):
        # right edge of left cell
        u0_l = e[k,:,1]
        # left edge of right cell
        u0_r = e[k+1,:,0]

        u0_avg = (u0_l + u0_r) / 2
        np.putmask(e[k,:,1],   u0_l != u0_r, u0_avg)
        np.putmask(e[k+1,:,0], u0_l != u0_r, u0_avg)

    # P1M constants
    c = np.empty_like(e)
    # x=0 value is left edge
    c[:,:,0] = e[:,:,0]
    # local slope given by difference of edge values
    c[:,:,1] = e[:,:,1] - e[:,:,0]

    # now we can actually perform the Newton-Raphson iteration
    # to find the interface positions corresponding with a target density
    # this is probably best implemented as a regular loop

    # loop over columns
    for j in range(h.shape[1]):
        # find the positions of all target values within the column
        # except the surface and very bottom
        for k, t in enumerate(targ_dens[1:-1]):
            # check whether we're too light, at an interface, or too dense
            if t <= e[0,j,0]:
                # too light, set to surface position
                z_new[k+1,j] = z[0,j]
                continue

            # do we land between the right edge of one cell
            # and the left edge of the next?
            # (in practice, this is just asking if we're exactly at the
            # interface because of the averaging we did before)
            i = (t >= e[:-1,j,1]) & (t <= e[1:,j,0])
            if np.any(i):
                z_new[k+1,j] = z[np.where(i)[0],j]
                continue

            if t >= e[-1,j,1]:
                # too dense, set to bottom position
                z_new[k+1,j] = z[-1,j]
                continue

            # we must be inside a cell, so find out which one
            i = (t > e[:,j,0]) & (t < e[:,j,1])
            ki = np.where(i)[0]

            # set up Newton-Raphson
            xi0 = 0.5
            i = 1
            delta = 1e10

            while i <= max_iter and abs(delta) >= max_tol:
                # polynomial at guess
                num = c[ki,j,0] + c[ki,j,1]*xi0 - t
                # gradient of interpolating function
                den = c[ki,j,1]

                delta = -num / den
                # update guess
                xi0 += delta

                # check whether new estimate is out of bounds
                if xi0 < 0:
                    xi0 = 0
                    if c[ki,j,1] == 0:
                        xi0 += eps

                if xi0 > 1:
                    xi0 = 1
                    if c[ki,j,1] == 0:
                        xi0 -= eps

                i += 1

            z_new[k+1,j] = z[ki,j] + xi0 * h[ki,j]

    # adjust new positions according to nominal depths
    z_nom = np.insert(dz_75.cumsum(), 0, 0)[:,np.newaxis]
    z_nom = np.tile(z_nom, (1, h.shape[1]))

    # regular hycom algorithm
    z_bnd = np.maximum(z_new, z_nom)
    # also bound by total depth
    z_bnd = np.minimum(z_bnd, z[[-1],:])

    # also also bound by maximum depth and thickness
    z_bnd[1:-1,:] = np.minimum(z_bnd[1:-1,:], max_int_depth[1:-1,np.newaxis],
                               z_bnd[:-2,:] + max_lay_thick[:-1,np.newaxis])

    # adjust nominal positions (transition pressure) based on salinity
    # scale by difference from some middle salinity
    z_nom_s = z_nom.copy()
    s_range = 0.5
    s0 = 35.0
    z_nom_s[1:,:] *= np.clip(1.0 - (sa - s0) / s_range, 0.5, 1.0)

    # enforce minimum dz
    for k in range(1, z_nom_s.shape[0] - 1):
        z_nom_s[k,:] = np.maximum(z_nom_s[k,:], z_nom_s[k-1,:] + 2)

    z_bnd_s = z_bnd.copy()

    # actual transition to nominal depth
    if s_rule:
        z_bnd_s = np.maximum(z_new, z_nom_s)

    if s_rule_alt:
        # alternate transition:
        # use non-modified positions for z when interface is too shallow
        # (we just want to be isopycnal for longer)
        z_bnd_s = np.where(z_new < z_nom_s, z_nom, z_new)

    if s_topo:
        # record the bottom interface of all layers
        z_shift = z_new.copy()
        z_shift[:-1,:] = z_new[1:,:]

        # transition to z when an isopycnal is shallower than its nominal depth
        # and, if this column is shallower than 500m, the bottom of the layer is
        # in the top 80% of the water column
        z_bnd_s = np.where((z_new < z_nom) & ((z_shift < 0.8 * z[-1,:]) | (z[-1,:] > 500)), z_nom, z_new)

    # detangle interfaces by pushing them upwards from below (i.e. isopycnal overrides)
    if detangle:
        for k in range(z_nom_s.shape[0] - 2, 0, -1):
            z_bnd_s[k,:] = np.minimum(z_bnd_s[k,:], z_bnd_s[k+1,:])


    # also bound by total depth
    z_bnd_s = np.minimum(z_bnd_s, z[-1,:])

    # also also bound by maximum depth and thickness
    z_bnd_s[1:-1,:] = np.minimum(z_bnd_s[1:-1,:], max_int_depth[1:-1,np.newaxis],
                                 z_bnd_s[:-2,:] + max_lay_thick[:-1,np.newaxis])

    return z_new, z_bnd, z_bnd_s
