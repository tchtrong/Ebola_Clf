def normalize(vec):

    temp_vec = vec
    ##temp_vec = [ele ** 0.15 for ele in temp_vec]
    sum_temp = sum(temp_vec)
    assert (abs(sum_temp) != 0.0)  # the sum must not be 0
    """
    if abs(s) < 1e-6:
        print "Sum of vectors sums almost to 0. Stop here."
        print "Vec: " + str(vec) + " Sum: " + str(s)
        assert(0) # assertion fails
    """

    for i in range(len(vec)):
        assert (vec[i] >= 0)  # element must be >= 0
        vec[i] = (vec[i]) / (sum_temp) #0.1 is beta-distribution for TEM algo
        #vec[i] = (vec[i]) ** 0.15 / (sum_temp) #0.1 is beta-distribution for TEM algo