#section support_code_struct

// Shorthand
#define DI0 DTYPE_INPUT_0
#define DI1 DTYPE_INPUT_1
#define DO0 DTYPE_OUTPUT_0
#define DO1 DTYPE_OUTPUT_1
#define DO2 DTYPE_OUTPUT_2
#define DO3 DTYPE_OUTPUT_3
#define DO4 DTYPE_OUTPUT_4
#define DO5 DTYPE_OUTPUT_5
#define TO0 TYPENUM_OUTPUT_0
#define TO1 TYPENUM_OUTPUT_1
#define TO2 TYPENUM_OUTPUT_2
#define TO3 TYPENUM_OUTPUT_3
#define TO4 TYPENUM_OUTPUT_4
#define TO5 TYPENUM_OUTPUT_5

int APPLY_SPECIFIC(latitude)(PyArrayObject *input0,   // alpha
                             PyArrayObject *input1,   // beta
                             PyArrayObject **output0, // q
                             PyArrayObject **output1, // dq / dalpha
                             PyArrayObject **output2, // dq / dbeta
                             PyArrayObject **output3, // Q
                             PyArrayObject **output4, // dQ / dalpha
                             PyArrayObject **output5  // dQ / dbeta
                             ) {

  using namespace sp::theano;
  using namespace sp::latitude;
  using namespace sp::utils;

  // Get the inputs
  int success = 0;
  int ndim = -1;
  npy_intp *shape;
  auto alpha_in = get_input<DI0>(&ndim, &shape, input0, &success);
  if (ndim != 0) {
    PyErr_Format(PyExc_ValueError, "alpha must be a scalar");
    return 1;
  }
  auto beta_in = get_input<DI1>(&ndim, &shape, input1, &success);
  if (ndim != 0) {
    PyErr_Format(PyExc_ValueError, "beta must be a scalar");
    return 1;
  }
  DI0 alpha = *alpha_in;
  DI1 beta = *beta_in;

  // Allocate the outputs
  int ndim_q = 1;
  int ndim_Q = 2;
  std::vector<npy_intp> shape_q_vec(ndim_q);
  shape_q_vec[0] = SP__N;
  npy_intp *shape_q = &(shape_q_vec[0]);
  std::vector<npy_intp> shape_Q_vec(ndim_Q);
  shape_Q_vec[0] = SP__N;
  shape_Q_vec[1] = SP__N;
  npy_intp *shape_Q = &(shape_Q_vec[0]);
  auto q_out = allocate_output<DO0>(ndim_q, shape_q, TO0, output0, &success);
  auto dqda_out = allocate_output<DO1>(ndim_q, shape_q, TO1, output1, &success);
  auto dqdb_out = allocate_output<DO2>(ndim_q, shape_q, TO2, output2, &success);
  auto Q_out = allocate_output<DO3>(ndim_Q, shape_Q, TO3, output3, &success);
  auto dQda_out = allocate_output<DO4>(ndim_Q, shape_Q, TO4, output4, &success);
  auto dQdb_out = allocate_output<DO5>(ndim_Q, shape_Q, TO5, output5, &success);
  if (success) {
    return 1;
  }
  Map<Vector<DO0, SP__N>> q(q_out);
  Map<Vector<DO1, SP__N>> dqda(dqda_out);
  Map<Vector<DO2, SP__N>> dqdb(dqdb_out);
  Map<RowMatrix<DO3, SP__N, SP__N>> Q(Q_out);
  Map<RowMatrix<DO4, SP__N, SP__N>> dQda(dQda_out);
  Map<RowMatrix<DO5, SP__N, SP__N>> dQdb(dQdb_out);

  // Compute the integrals
  computeLatitudeIntegrals(alpha, beta, q, dqda, dqdb, Q, dQda, dQdb);

  // We're done!
  return 0;
}
