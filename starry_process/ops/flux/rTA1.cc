#section support_code_struct

// Shorthand
#define DO0 DTYPE_OUTPUT_0
#define TO0 TYPENUM_OUTPUT_0

int APPLY_SPECIFIC(rTA1)(PyArrayObject **output0) {

  using namespace sp::theano;
  using namespace sp::flux;
  using namespace sp::utils;

  // Allocate the outputs
  int success = 0;
  int ndim = 1;
  std::vector<npy_intp> shape_vec(ndim);
  shape_vec[0] = SP__N;
  npy_intp *shape = &(shape_vec[0]);
  auto f_out = allocate_output<DO0>(ndim, shape, TO0, output0, &success);
  if (success) {
    return 1;
  }
  Map<Vector<DO0, SP__N>> f(f_out);

  // Compute!
  computerTA1(f);

  // We're done!
  return 0;
}
