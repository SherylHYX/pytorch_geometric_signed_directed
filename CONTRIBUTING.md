# How to contribute

Glad that you are reading this, as we welcome your contributions to make this project better!

Here are some important things to check when you contribute:

  * Please make sure that you write tests.
  * Update the documentation.
  * Add the new model to the readme.
  * It would be great if you could also add an example using the model in the `examples/` folder.
  
## Testing


PyTorch Geometric Signed Directed's testing is located under `test/`.
Run the entire test suite with

```
pytest
```

## Submitting changes

Please send a [GitHub Pull Request to PyTorch Geometric Signed Directed](https://github.com/SherylHYX/pytorch_geometric_signed_directed/pulls) with a clear list of what you've done (read more about [pull requests](http://help.github.com/pull-requests/)). Please follow our coding conventions (below) and make sure all of your commits are atomic (one feature per commit).

Always write a clear log message for your commits. One-line messages are fine for small changes, but bigger changes should look like this:

    $ git commit -m "A brief summary of the commit
    > 
    > A paragraph describing what changed and its impact."

## Coding conventions

Start reading our code and you'll get the hang of it. We optimize for readability:

  * We write tests for the data loaders, utilities and layers.
  * We use the type hinting feature of Python.

Sincerely,
Yixuan He

