from starry_process import calibrate


def test_calibrate():
    # Very bare-bones test for calibration runs
    # Just checking that no errors are thrown!
    kwargs = dict(
        generate=dict(nlc=5),
        sample=dict(run_nested_kwargs=dict(maxiter=10)),
        plot=dict(ninc_points=5, ninc_samples=2),
    )
    calibrate.run(path=".test_calibrate", **kwargs)
