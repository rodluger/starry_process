#section support_code_struct

// Shorthand
#define DI0 DTYPE_INPUT_0
#define DO0 DTYPE_OUTPUT_0
#define DO1 DTYPE_OUTPUT_1
#define TO0 TYPENUM_OUTPUT_0
#define TO1 TYPENUM_OUTPUT_1

int APPLY_SPECIFIC(Ry)(PyArrayObject *input0,   // theta
                       PyArrayObject **output0, // R
                       PyArrayObject **output1  // dR / dtheta
                       ) {

  using namespace sp::theano;
  using namespace sp::wigner;
  using namespace sp::utils;

  // Get the inputs
  int success = 0;
  int ndim = -1;
  npy_intp *shape;
  auto theta_in = get_input<DI0>(&ndim, &shape, input0, &success);
  if (ndim != 0) {
    PyErr_Format(PyExc_ValueError, "theta must be a scalar");
    return 1;
  }
  DI0 theta = *theta_in;

  // Allocate the outputs
  ndim = 1;
  std::vector<npy_intp> shape_vec(ndim);
  shape_vec[0] = SP__NWIG;
  shape = &(shape_vec[0]);
  auto Ry_out = allocate_output<DO0>(ndim, shape, TO0, output0, &success);
  auto dRydtheta_out =
      allocate_output<DO1>(ndim, shape, TO1, output1, &success);
  if (success) {
    return 1;
  }
  Map<Vector<DO0, SP__NWIG>> Ry(Ry_out);
  Map<Vector<DO1, SP__NWIG>> dRydtheta(dRydtheta_out);

  // Compute!
  computeRy(theta, Ry, dRydtheta);

  // We're done!
  return 0;
}
