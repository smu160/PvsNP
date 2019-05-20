"""Test script that runs the Demo notebook."""


import subprocess
import tempfile
import nbformat


def _notebook_run(path):
    """Execute a notebook via nbconvert and collect output.

    Parameters
    ----------
    path:

    Returns
    -------
    nb, errors: tuple
        A tuple of the parsed nb object and the execution errors.
    """
    kernel = "python3"

    with tempfile.NamedTemporaryFile(suffix=".ipynb") as fout:
        args = ["jupyter", "nbconvert", "--to", "notebook", "--execute",
                "--ExecutePreprocessor.timeout=1200",
                "--ExecutePreprocessor.kernel_name="+kernel,
                "--output", fout.name, path]

        subprocess.check_call(args)

        fout.seek(0)
        nb = nbformat.reads(fout.read().decode('utf-8'), nbformat.current_nbformat)

    errors = [output for cell in nb.cells if "outputs" in cell
              for output in cell["outputs"]
              if output.output_type == "error"]

    return nb, errors


def run_tests(*notebooks):
    """Run the test on every provided notebook

    Parameters
    ----------
    notebooks: str
        The full filepaths to the notebooks to be tested.
    """
    for filename in notebooks:
        if filename.split('.')[-1] == "ipynb":
            _, errors = _notebook_run(filename)
            if errors != []:
                raise Exception


def main():
    """Begin the test script here"""
    notebooks = ["examples/Demo.ipynb"]
    run_tests(*notebooks)


if __name__ == "__main__":
    main()
