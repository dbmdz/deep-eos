# Coding style

*deep-eos* uses the [PEP 8](https://www.python.org/dev/peps/pep-0008/) coding
conventions. Thus [`autopep8`](https://pypi.python.org/pypi/autopep8) is used
for automatically formatting source files.

## Installing `autopep8`

`autopep8` can be installed via `pip`:

```bash
pip3 install --user autopep8
```

## Format source files

To automatically formatting a source file, just execute the following command:

```
autopep8 --in-place --aggressive --aggressive <filename>
```

**Notice**: This will in-place edit the file!

# Documentation

All functions should be well-documented. As documentation style we use the Keras
documentation style, e.g. a function can be documented like:

```python
def save_model(model, filepath, overwrite=True, include_optimizer=True):
    """Save a model to a HDF5 file.
    The saved model contains:
        - the model's configuration (topology)
        - the model's weights
        - the model's optimizer's state (if any)
    Thus the saved model can be reinstantiated in
    the exact same state, without any of the code
    used for model definition or training.
    # Arguments
        model: Keras model instance to be saved.
        filepath: String, path where to save the model.
        overwrite: Whether we should overwrite any existing
            model at the target location, or instead
            ask the user with a manual prompt.
        include_optimizer: If True, save optimizer's state together.
    # Raises
        ImportError: if h5py is not available.
    """
```

All arguments or return values should be descibed clearly.
