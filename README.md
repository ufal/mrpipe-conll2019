# Sources from ÚFAL MRPipe at MRP 2019 paper

This repository contains the raw sources from the
**ÚFAL MRPipe at MRP 2019: UDPipe Goes Semantic in the Meaning Representation Parsing Shared Task**
paper.

The sources are available under MPL licenses, unless specified otherwise.

The repository contains the exact sources used for the shared task. Running
the code requires the original data, fastText embeddings and Bert multilingual
checkpoint. After creating all required data in `generated` subdirectory, you
can start training by running `run.sh` in the `src` directory, which generates
a file with five lines, each a command running `mr_pipe.py` for training one of
the frameworks.
