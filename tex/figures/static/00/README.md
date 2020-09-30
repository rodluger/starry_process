This worked surprisingly well. We successfully inferred the  latitude and 
contrast params, as well as all of the inclinations. There was quite a bit
of bias in the spot size params, but that's expected: there's no reason that
the radius of a discrete circular spot should be the same as the HWHM of
a Lorentzian spot -- there should naturally be some scaling we need to
calibrate.

Note also that I ran a NUTS chain for about 60 hours (!) and it agreed
extremely well with the ADVI results. Even after 60 hours, the effective
sample size was still only about 200 for some of the parameters, so the
posteriors were quite noisy. ADVI seems to be the way to go!