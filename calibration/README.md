To run a batch simulation:
```
name = "default"
kwargs = {}
python -c "from calibration import run_batch; run_batch(name, **kwargs)"
```

Once that's done, run the postprocessing:
```
name = "default"
python -c "from calibration import process_batch; process_batch(name)"
```

Finally, plot the results:
```
name = "default"
python -c "from calibration import plot_batch; plot_batch(name)"
```