#section support_code_struct

// Shorthand
#define DI0 DTYPE_INPUT_0
#define DI1 DTYPE_INPUT_1
#define DI2 DTYPE_INPUT_2
#define DI3 DTYPE_INPUT_3
#define DI4 DTYPE_INPUT_4
#define DO0 DTYPE_OUTPUT_0
#define TO0 TYPENUM_OUTPUT_0

int APPLY_SPECIFIC(eigh)(PyArrayObject *input0,  // x
                         PyArrayObject *input1,  // w
                         PyArrayObject *input2,  // v
                         PyArrayObject *input3,  // gw
                         PyArrayObject *input4,  // gv
                         PyArrayObject **output0 // bx
                         ) {

  using namespace sp::theano;
  using namespace sp::utils;
  using namespace sp::eigh;

  // Get the inputs
  int success = 0;
  int ndim0 = -1;
  npy_intp *shape0;
  auto x_in = get_input<DI0>(&ndim0, &shape0, input0, &success);
  if (ndim0 != 2) {
    PyErr_Format(PyExc_ValueError, "x must be a matrix");
    return 1;
  }
  if (success) {
    return 1;
  }
  int M = shape0[0];
  int ndim1 = -1;
  npy_intp *shape1;
  auto w_in = get_input<DI1>(&ndim1, &shape1, input1, &success);
  if (ndim1 != 1) {
    PyErr_Format(PyExc_ValueError, "w must be a vector");
    return 1;
  }
  if (success) {
    return 1;
  }
  int N = shape1[0];
  int ndim2 = -1;
  npy_intp *shape2;
  auto v_in = get_input<DI2>(&ndim2, &shape2, input2, &success);
  if (ndim2 != 2) {
    PyErr_Format(PyExc_ValueError, "v must be a matrix");
    return 1;
  }
  if (success) {
    return 1;
  }
  int ndim3 = -1;
  npy_intp *shape3;
  auto gw_in = get_input<DI3>(&ndim3, &shape3, input3, &success);
  if (ndim3 != 1) {
    PyErr_Format(PyExc_ValueError, "gw must be a vector");
    return 1;
  }
  if (success) {
    return 1;
  }
  int ndim4 = -1;
  npy_intp *shape4;
  auto gv_in = get_input<DI4>(&ndim4, &shape4, input4, &success);
  if (ndim4 != 2) {
    PyErr_Format(PyExc_ValueError, "gv must be a matrix");
    return 1;
  }
  if (success) {
    return 1;
  }
  Map<RowMatrix<DI0, Dynamic, Dynamic>> x(x_in, M, M);
  Map<Vector<DI1, Dynamic>> w(w_in, N);
  Map<RowMatrix<DI2, Dynamic, Dynamic>> v(v_in, M, N);
  Map<Vector<DI3, Dynamic>> gw(gw_in, N);
  Map<RowMatrix<DI4, Dynamic, Dynamic>> gv(gv_in, M, N);

  // Allocate the outputs
  int ndim = 2;
  std::vector<npy_intp> shape_vec(ndim);
  shape_vec[0] = M;
  shape_vec[1] = M;
  npy_intp *shape = &(shape_vec[0]);
  auto gx_out = allocate_output<DO0>(ndim, shape, TO0, output0, &success);
  if (success) {
    return 1;
  }
  Map<RowMatrix<DI0, Dynamic, Dynamic>> gx(gx_out, M, M);

  // Compute!
  eigh_grad(x, w, v, gw, gv, gx);

  // We're done!
  return 0;
}
