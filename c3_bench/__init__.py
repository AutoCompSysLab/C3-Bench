"""C3-Bench: a change-captioning benchmark for vision-language models.

Modules
-------
* :mod:`c3_bench.infer`   - run a VLM over image pairs to produce change captions.
* :mod:`c3_bench.eval`    - score predictions against references with an LLM judge.
* :mod:`c3_bench.prompts` - per-domain change criteria and prompt templates.
* :mod:`c3_bench.models`  - unified model loading and generation helpers.
* :mod:`c3_bench.io_utils`, :mod:`c3_bench.dist_utils` - shared utilities.
"""

__version__ = "0.1.0"
