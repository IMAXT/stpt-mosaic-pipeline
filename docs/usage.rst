Usage
=====

In order to run the pipeline in the IMAXT servers you will need a pipeline
definition file and the :ref:`Owl Client <imaxt:owl_client>` to submit the job.

.. _stpt_pipedef:

Pipeline definition file
-------------------------

In order to run a pipeline we need a configuration file that sets the inputs,
outputs and relevant paramenters needed for the various steps.

An example pipeline definition file follows:

.. code-block:: yaml

    # Version of the configuration file
    version: 1.1

    # Name of the pipeline
    extra_pip_packages: stpt-pipeline
    name: stpt

    # I/O
    root_dir: /data/meds1_a/imaxtapp/incoming/MA1-0003/STPT/MA1-191127-1125
    output_dir: /data/meds1_b/processed/STPT

    # Recipes to run
    recipes: ["distortion", "mosaic", "downsample", "beadreg"]

    # Resources requested
    resources:
      threads: 1
      workers: 20
      memory: 12


General options
'''''''''''''''

+------------------------+----------------------------------------------------+
| **version**            | Used to mark different versions of the pipeline    |
|                        | configuration format.                              |
+------------------------+----------------------------------------------------+
| **extra_pip_packages** | Packages required to run the pipeline. This should |
|                        | be set to only the package containing the pipeline |
+------------------------+----------------------------------------------------+
| **name**               | Name of the pipeline as defined in the package     |
|                        | above.                                             |
+------------------------+----------------------------------------------------+
| **root_dir**           | Location of input data in OME-TIFF format.         |
+------------------------+----------------------------------------------------+
| **output_dir**         | Location to write outputs to. The output directory |
|                        | will be named as as the sample name inside         |
|                        | ``output_dir``                                     |
+------------------------+----------------------------------------------------+
| **recipes**            | Recipes to run (see below)                         |
+------------------------+----------------------------------------------------+


Available recipes
'''''''''''''''''

+------------------------+----------------------------------------------------+
| **distortion**         | Performs distortion correction as well as dark and |
|                        | flatfield correction.                              |
+------------------------+----------------------------------------------------+
| **mosaic**             | Performs the mosaic of tiles.                      |
+------------------------+----------------------------------------------------+
| **downsample**         | Downsamples the mosaic to various resolution       |
|                        | levels.                                            |
+------------------------+----------------------------------------------------+
| **beadreg**            | Perform bead registration                          |
+------------------------+----------------------------------------------------+


Resources section
'''''''''''''''''

+------------------------+----------------------------------------------------+
| **workers**            | Number of independent workers to launch.           |
+------------------------+----------------------------------------------------+
| **threads**            | Number of independent threads per worker.          |
+------------------------+----------------------------------------------------+
| **memory**             | Memory per worker in GB.                           |
+------------------------+----------------------------------------------------+

.. _stpt_running:

Running the pipeline
--------------------

In order to run the pipeline ensure that the
:ref:`Owl Client <imaxt:owl_client>`
is installed in your Python environment and follow the
:ref:`usage instructions <imaxt:owl_client_usage>`.
