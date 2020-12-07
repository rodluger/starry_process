#section support_code_struct

// Shorthand
#define DI0 DTYPE_INPUT_0
#define DI1 DTYPE_INPUT_1
#define DO0 DTYPE_OUTPUT_0
#define TO0 TYPENUM_OUTPUT_0

sp::flux::LimbDark<DO0> *APPLY_SPECIFIC(LD_rev);

#section init_code_struct

{ APPLY_SPECIFIC(LD_rev) = NULL; }

#section cleanup_code_struct

if (APPLY_SPECIFIC(LD_rev) != NULL) {
  delete APPLY_SPECIFIC(LD_rev);
}

#section support_code_struct

int APPLY_SPECIFIC(rTA1L_rev)(PyArrayObject *input0, PyArrayObject *input1,
                              PyArrayObject **output0) {

  using namespace sp::theano;
  using namespace sp::flux;
  using namespace sp::utils;

  // Get the inputs
  int success = 0;
  int ndim = -1;
  npy_intp *shape;
  auto u_in = get_input<DI0>(&ndim, &shape, input0, &success);
  if (ndim != 1) {
    PyErr_Format(PyExc_ValueError, "u must be a vector");
    return 1;
  }
  Map<Vector<DI0, SP__UMAX>> u(u_in);

  ndim = -1;
  auto bf_in = get_input<DI1>(&ndim, &shape, input1, &success);
  if (ndim != 1) {
    PyErr_Format(PyExc_ValueError, "bf must be a vector");
    return 1;
  }
  Map<RowVector<DI1, SP__N>> bf(bf_in);

  // Allocate the outputs
  ndim = 1;
  std::vector<npy_intp> shape_vec(ndim);
  shape_vec[0] = SP__UMAX;
  shape = &(shape_vec[0]);
  auto bu_out = allocate_output<DO0>(ndim, shape, TO0, output0, &success);
  if (success) {
    return 1;
  }
  Map<Vector<DO0, SP__UMAX>> bu(bu_out);

  // Initialize the class if needed
  if (APPLY_SPECIFIC(LD_rev) == NULL) {
    APPLY_SPECIFIC(LD_rev) = new LimbDark<DO0>();
  }

  // Compute
  APPLY_SPECIFIC(LD_rev)->template computerTA1L(u, bf, bu);

  // We're done!
  return 0;
}
