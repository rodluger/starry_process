#section support_code_struct

// Shorthand
#define DI0 DTYPE_INPUT_0
#define DI1 DTYPE_INPUT_1
#define DO0 DTYPE_OUTPUT_0
#define DO1 DTYPE_OUTPUT_1
#define DO2 DTYPE_OUTPUT_2
#define TO0 TYPENUM_OUTPUT_0
#define TO1 TYPENUM_OUTPUT_1
#define TO2 TYPENUM_OUTPUT_2

int APPLY_SPECIFIC(tensordotRz)(PyArrayObject *input0,   // M
                                PyArrayObject *input1,   // theta
                                PyArrayObject **output0, // f(M)
                                PyArrayObject **output1, // d f(M) / dM
                                PyArrayObject **output2  // d f(M) / dtheta
                                ) {

  using namespace sp::theano;
  using namespace sp::wigner;
  using namespace sp::utils;

  // Get the inputs
  int success = 0;
  int ndim = -1;
  npy_intp *shape;
  auto M_in = get_input<DI0>(&ndim, &shape, input0, &success);
  if (ndim != 2) {
    PyErr_Format(PyExc_ValueError, "M must be a matrix");
    return 1;
  }
  ndim = -1;
  auto theta_in = get_input<DI1>(&ndim, &shape, input1, &success);
  if (ndim != 1) {
    PyErr_Format(PyExc_ValueError, "theta must be a vector");
    return 1;
  }
  int K = shape[0];
  Map<RowMatrix<DO0, Dynamic, SP__N>> M(M_in, K, SP__N);
  Map<Vector<DO1, Dynamic>> theta(theta_in, K);

  // Allocate the outputs
  ndim = 2;
  std::vector<npy_intp> shape_KxN_vec(ndim);
  shape_KxN_vec[0] = K;
  shape_KxN_vec[1] = SP__N;
  npy_intp *shape_KxN = &(shape_KxN_vec[0]);
  std::vector<npy_intp> shape_NxN_vec(ndim);
  shape_NxN_vec[0] = SP__N;
  shape_NxN_vec[1] = SP__N;
  npy_intp *shape_NxN = &(shape_NxN_vec[0]);
  auto f_out = allocate_output<DO0>(ndim, shape_KxN, TO0, output0, &success);
  auto dfdM_out = allocate_output<DO1>(ndim, shape_NxN, TO1, output1, &success);
  auto dfdtheta_out =
      allocate_output<DO2>(ndim, shape_KxN, TO2, output2, &success);
  if (success) {
    return 1;
  }
  Map<RowMatrix<DO0, Dynamic, SP__N>> f(f_out, K, SP__N);
  Map<RowMatrix<DO1, SP__N, SP__N>> dfdM(dfdM_out);
  Map<RowMatrix<DO2, Dynamic, SP__N>> dfdtheta(dfdtheta_out, K, SP__N);

  // Compute!
  computeTensordotRz(M, theta, f, dfdM, dfdtheta);

  // We're done!
  return 0;
}
