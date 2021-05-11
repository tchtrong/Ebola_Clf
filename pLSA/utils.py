def normalize(vec):
    s = sum(vec)
    assert (abs(s) != 0.0)  # the sum must not be 0
    """
    if abs(s) < 1e-6:
        print "Sum of vectors sums almost to 0. Stop here."
        print "Vec: " + str(vec) + " Sum: " + str(s)
        assert(0) # assertion fails
    """

    for i in range(len(vec)):
        assert (vec[i] >= 0)  # element must be >= 0
        vec[i] = vec[i] * 1.0 / s