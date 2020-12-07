#section support_code_struct

// Shorthand
#define DI0 DTYPE_INPUT_0
#define DO0 DTYPE_OUTPUT_0
#define TO0 TYPENUM_OUTPUT_0

sp::flux::LimbDark<DO0> *APPLY_SPECIFIC(LD);

#section init_code_struct

{ APPLY_SPECIFIC(LD) = NULL; }

#section cleanup_code_struct

if (APPLY_SPECIFIC(LD) != NULL) {
  delete APPLY_SPECIFIC(LD);
}

#section support_code_struct

int APPLY_SPECIFIC(rTA1L)(PyArrayObject *input0, PyArrayObject **output0) {

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
  Map<Vector<DO0, SP__UMAX>> u(u_in);

  // Allocate the outputs
  ndim = 1;
  std::vector<npy_intp> shape_vec(ndim);
  shape_vec[0] = SP__N;
  shape = &(shape_vec[0]);
  auto f_out = allocate_output<DO0>(ndim, shape, TO0, output0, &success);
  if (success) {
    return 1;
  }
  Map<RowVector<DO0, SP__N>> f(f_out);

  // Initialize the class if needed
  if (APPLY_SPECIFIC(LD) == NULL) {
    APPLY_SPECIFIC(LD) = new LimbDark<DO0>();
  }

  // Compute
  APPLY_SPECIFIC(LD)->template computerTA1L(u, f);

  // We're done!
  return 0;
}
