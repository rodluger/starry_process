#section support_code_struct

// Shorthand
#define DI0 DTYPE_INPUT_0
#define DI1 DTYPE_INPUT_1
#define DI2 DTYPE_INPUT_2
#define DO0 DTYPE_OUTPUT_0
#define TO0 TYPENUM_OUTPUT_0

Eigen::SparseMatrix<DO0> *APPLY_SPECIFIC(A1);

#section init_code_struct

{ APPLY_SPECIFIC(A1) = NULL; }

#section cleanup_code_struct

if (APPLY_SPECIFIC(A1) != NULL) {
  delete APPLY_SPECIFIC(A1);
}

#section support_code_struct

int APPLY_SPECIFIC(pTA1)(PyArrayObject *input0, // x
                         PyArrayObject *input1, // y
                         PyArrayObject *input2, // z
                         PyArrayObject **output0) {

  using namespace sp::theano;
  using namespace sp::flux;
  using namespace sp::utils;

  // Get the inputs
  int success = 0;
  int ndim = -1;
  npy_intp *shape;
  auto x_in = get_input<DI0>(&ndim, &shape, input0, &success);
  if (ndim != 1) {
    PyErr_Format(PyExc_ValueError, "x must be a vector");
    return 1;
  }
  auto y_in = get_input<DI1>(&ndim, &shape, input1, &success);
  if (ndim != 1) {
    PyErr_Format(PyExc_ValueError, "y must be a vector");
    return 1;
  }
  auto z_in = get_input<DI2>(&ndim, &shape, input2, &success);
  if (ndim != 1) {
    PyErr_Format(PyExc_ValueError, "z must be a vector");
    return 1;
  }
  int K = shape[0];
  Map<RowVector<DI0, Dynamic>> x(x_in, K);
  Map<RowVector<DI0, Dynamic>> y(y_in, K);
  Map<RowVector<DI0, Dynamic>> z(z_in, K);

  // Allocate the outputs
  success = 0;
  ndim = 2;
  std::vector<npy_intp> shape_vec(ndim);
  shape_vec[0] = K;
  shape_vec[1] = SP__N;
  npy_intp *shape_pTA1 = &(shape_vec[0]);
  auto pTA1_out =
      allocate_output<DO0>(ndim, shape_pTA1, TO0, output0, &success);
  if (success) {
    return 1;
  }
  Map<RowMatrix<DO0, Dynamic, Dynamic>> pTA1(pTA1_out, K, SP__N);

  // Compute the `pT` vector
  computepT(x, y, z, pTA1);

  // Change to Ylm basis
  if (APPLY_SPECIFIC(A1) == NULL) {
    APPLY_SPECIFIC(A1) = new Eigen::SparseMatrix<DO0>();
    computeA1(SP__LMAX, *APPLY_SPECIFIC(A1));
  }
  pTA1 = pTA1 * *APPLY_SPECIFIC(A1);

  // We're done!
  return 0;
}
