#section support_code_struct

// Shorthand
#define DI0 DTYPE_INPUT_0
#define DI1 DTYPE_INPUT_1
#define DI2 DTYPE_INPUT_2
#define DO0 DTYPE_OUTPUT_0
#define TO0 TYPENUM_OUTPUT_0

int APPLY_SPECIFIC(special_tensordotRz)(
    PyArrayObject *input0,  // T
    PyArrayObject *input1,  // M
    PyArrayObject *input2,  // theta
    PyArrayObject **output0 // (something like) T_ij R_ilk M_lj
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
  int K = shape[0];
  Map<RowMatrix<DI0, SP__N, SP__N>> T(T_in);
  Map<RowMatrix<DI1, SP__N, SP__N>> M(M_in);
  Map<Vector<DI2, Dynamic>> theta(theta_in, K);

  // Allocate the outputs
  ndim = 1;
  std::vector<npy_intp> shape_vec(ndim);
  shape_vec[0] = K;
  shape = &(shape_vec[0]);
  auto f_out = allocate_output<DO0>(ndim, shape, TO0, output0, &success);
  if (success) {
    return 1;
  }
  Map<Vector<DO0, Dynamic>> f(f_out, K);

  // Compute!
  computeSpecialTensordotRz(T, M, theta, f);

  // We're done!
  return 0;
}
