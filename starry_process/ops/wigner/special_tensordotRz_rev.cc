#section support_code_struct

// Shorthand
#define DI0 DTYPE_INPUT_0
#define DI1 DTYPE_INPUT_1
#define DI2 DTYPE_INPUT_2
#define DI3 DTYPE_INPUT_3
#define DO0 DTYPE_OUTPUT_0
#define DO1 DTYPE_OUTPUT_1
#define TO0 TYPENUM_OUTPUT_0
#define TO1 TYPENUM_OUTPUT_1

int APPLY_SPECIFIC(special_tensordotRz_rev)(PyArrayObject *input0,   // T
                                            PyArrayObject *input1,   // M
                                            PyArrayObject *input2,   // theta
                                            PyArrayObject *input3,   // bf
                                            PyArrayObject **output0, // bM
                                            PyArrayObject **output1  // btheta
                                            ) {

  using namespace sp::theano;
  using namespace sp::wigner;
  using namespace sp::utils;

  // Get the inputs
  int success = 0;
  int ndim = -1;
  npy_intp *shape;
  auto T_in = get_input<DI0>(&ndim, &shape, input0, &success);
  if (ndim != 2) {
    PyErr_Format(PyExc_ValueError, "T must be a matrix");
    return 1;
  }
  auto M_in = get_input<DI1>(&ndim, &shape, input1, &success);
  if (ndim != 2) {
    PyErr_Format(PyExc_ValueError, "M must be a matrix");
    return 1;
  }
  ndim = -1;
  auto theta_in = get_input<DI2>(&ndim, &shape, input2, &success);
  if (ndim != 1) {
    PyErr_Format(PyExc_ValueError, "theta must be a vector");
    return 1;
  }
  ndim = -1;
  auto bf_in = get_input<DI3>(&ndim, &shape, input3, &success);
  if (ndim != 1) {
    PyErr_Format(PyExc_ValueError, "bf must be a vector");
    return 1;
  }
  int K = shape[0];
  Map<RowMatrix<DI0, SP__N, SP__N>> T(T_in);
  Map<RowMatrix<DI1, SP__N, SP__N>> M(M_in);
  Map<Vector<DI2, Dynamic>> theta(theta_in, K);
  Map<Vector<DI3, Dynamic>> bf(bf_in, K);

  // Allocate the outputs
  ndim = 2;
  std::vector<npy_intp> shape_NxN_vec(ndim);
  shape_NxN_vec[0] = SP__N;
  shape_NxN_vec[1] = SP__N;
  npy_intp *shape_NxN = &(shape_NxN_vec[0]);
  auto bM_out = allocate_output<DO0>(ndim, shape_NxN, TO0, output0, &success);
  ndim = 1;
  std::vector<npy_intp> shape_K_vec(ndim);
  shape_K_vec[0] = K;
  npy_intp *shape_K = &(shape_K_vec[0]);
  auto btheta_out = allocate_output<DO1>(ndim, shape_K, TO1, output1, &success);
  if (success) {
    return 1;
  }
  Map<RowMatrix<DO0, SP__N, SP__N>> bM(bM_out);
  Map<Vector<DO1, Dynamic>> btheta(btheta_out, K);

  // Compute!
  computeSpecialTensordotRzGradient(T, M, theta, bf, bM, btheta);

  // We're done!
  return 0;
}
