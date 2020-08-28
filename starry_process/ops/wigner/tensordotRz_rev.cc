#section support_code_struct

// Shorthand
#define DI0 DTYPE_INPUT_0
#define DI1 DTYPE_INPUT_1
#define DI2 DTYPE_INPUT_2
#define DO0 DTYPE_OUTPUT_0
#define DO1 DTYPE_OUTPUT_1
#define TO0 TYPENUM_OUTPUT_0
#define TO1 TYPENUM_OUTPUT_1

int APPLY_SPECIFIC(tensordotRz_rev)(PyArrayObject *input0,   // M
                                    PyArrayObject *input1,   // theta
                                    PyArrayObject *input2,   // bf
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
  ndim = -1;
  auto bf_in = get_input<DI2>(&ndim, &shape, input2, &success);
  if (ndim != 2) {
    PyErr_Format(PyExc_ValueError, "bf must be a matrix");
    return 1;
  }
  int K = shape[0];
  Map<RowMatrix<DI0, Dynamic, SP__N>> M(M_in, K, SP__N);
  Map<Vector<DI1, Dynamic>> theta(theta_in, K);
  Map<RowMatrix<DI2, Dynamic, SP__N>> bf(bf_in, K, SP__N);

  // Allocate the outputs
  ndim = 2;
  std::vector<npy_intp> shape_KxN_vec(ndim);
  shape_KxN_vec[0] = K;
  shape_KxN_vec[1] = SP__N;
  npy_intp *shape_KxN = &(shape_KxN_vec[0]);
  auto bM_out = allocate_output<DO0>(ndim, shape_KxN, TO0, output0, &success);
  ndim = 1;
  std::vector<npy_intp> shape_K_vec(ndim);
  shape_K_vec[0] = K;
  npy_intp *shape_K = &(shape_K_vec[0]);
  auto btheta_out = allocate_output<DO1>(ndim, shape_K, TO1, output1, &success);
  if (success) {
    return 1;
  }
  Map<RowMatrix<DO0, Dynamic, SP__N>> bM(bM_out, K, SP__N);
  Map<Vector<DO1, Dynamic>> btheta(btheta_out, K);

  // Compute!
  computeTensordotRzGradient(M, theta, bf, bM, btheta);

  // We're done!
  return 0;
}
