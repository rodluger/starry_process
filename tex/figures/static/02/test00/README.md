This run again did not go well. Once again, the inclinations were all
correct, but the size, latitude, and contrast hyperparameters are
incorrect. As before, the posterior map samples tell a different story:
there is strong evidence for equatorial spots! They are, however, 
significantly smaller than in the true maps.

Again, it's possible that the best course of action is to draw posterior
map samples and compute statistics based on those.

Also, part of the reason for the incorrect result is that there are simply
too many spots in the true maps. They overlap each other so much that they
form an equatorial band. This band is entirely in the null space, so all we
can measure are deviations from it. I'm going to re-run this with fewer spots,
which is probably more realistic anyways!