#section support_code_struct

// starry::limbdark::GreensLimbDark<DTYPE_OUTPUT_0> *APPLY_SPECIFIC(L);

#section init_code_struct

//{ APPLY_SPECIFIC(L) = NULL; }

#section cleanup_code_struct

// if (APPLY_SPECIFIC(L) != NULL) {
//  delete APPLY_SPECIFIC(L);
//}

#section support_code_struct

int APPLY_SPECIFIC(latitude)(PyArrayObject *input0,   // alpha
                             PyArrayObject *input1,   // beta
                             PyArrayObject **output0, // q
                             PyArrayObject **output1, // dq / dalpha
                             PyArrayObject **output2, // dq / dbeta
                             PyArrayObject **output3, // Q
                             PyArrayObject **output4, // dQ / dalpha
                             PyArrayObject **output5  // dQ / dbeta
                             ) {
  // using namespace starry_process;

  using Eigen::Map;
  using Eigen::Matrix;
  using Eigen::RowMajor;

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

  // DEBUG, TODO?
  const npy_intp ydeg = 15;
  const npy_intp N = (ydeg + 1) * (ydeg + 1);

  // Get the inputs
  int success = 0;
  int ndim = -1;
  npy_intp *shape;
  auto alpha = get_input<DI0>(&ndim, &shape, input0, &success);
  if (ndim != 0) {
    PyErr_Format(PyExc_ValueError, "alpha must be a scalar");
    return 1;
  }
  auto beta = get_input<DI1>(&ndim, &shape, input1, &success);
  if (ndim != 0) {
    PyErr_Format(PyExc_ValueError, "beta must be a scalar");
    return 1;
  }

  // Allocate the outputs
  std::vector<npy_intp> shape_q(1);
  shape_q[0] = N;
  std::vector<npy_intp> shape_Q(2);
  shape_Q[0] = N;
  shape_Q[1] = N;
  auto q_out = allocate_output<DO0>(1, shape_q, TO0, output0, &success);
  auto dqda_out = allocate_output<DO1>(1, shape_q, TO1, output1, &success);
  auto dqdb_out = allocate_output<DO2>(1, shape_q, TO2, output2, &success);
  auto Q_out = allocate_output<DO3>(2, shape_Q, TO3, output3, &success);
  auto dQda_out = allocate_output<DO4>(2, shape_Q, TO4, output4, &success);
  auto dQdb_out = allocate_output<DO5>(2, shape_Q, TO5, output5, &success);
  if (success)
    return 1;
  Map<Matrix<DO0, N, 1>> q(q_out, N);
  Map<Matrix<DO1, N, 1>> dqda(dqda_out, N);
  Map<Matrix<DO2, N, 1>> dqdb(dqdb_out, N);
  Map<Matrix<DO3, N, N, RowMajor>> Q(Q_out, N, N);
  Map<Matrix<DO4, N, N, RowMajor>> dQda(dQda_out, N, N);
  Map<Matrix<DO5, N, N, RowMajor>> dQdb(dQdb_out, N, N);

  q.setZero();
  dqda.setZero();
  dqdb.setZero();
  Q.setZero();
  dQda.setZero();
  dQdb.setZero();

  /*
  if (APPLY_SPECIFIC(L) == NULL || APPLY_SPECIFIC(L)->lmax != Nc - 1) {
    if (APPLY_SPECIFIC(L) != NULL)
      delete APPLY_SPECIFIC(L);
    APPLY_SPECIFIC(L) = new starry::limbdark::GreensLimbDark<double>(Nc - 1);
  }

  for (npy_intp i = 0; i < Nb; ++i) {
    f[i] = 0;
    dfdb[i] = 0;
    dfdr[i] = 0;

    if (los[i] > 0) {
      auto b_ = std::abs(b[i]);
      auto r_ = std::abs(r[i]);
      if (b_ < 1 + r_) {
        APPLY_SPECIFIC(L)->template compute<true>(b_, r_);
        auto sT = APPLY_SPECIFIC(L)->sT;

        // The value of the light curve
        f[i] = sT.dot(cvec);

        // The gradients
        dfdcl_mat.col(i) = sT;
        dfdb[i] = sgn(b[i]) * APPLY_SPECIFIC(L)->dsTdb.dot(cvec);
        dfdr[i] = sgn(r[i]) * APPLY_SPECIFIC(L)->dsTdr.dot(cvec);
      }
    }
  }
  */

  return 0;
}
