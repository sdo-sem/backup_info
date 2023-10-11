import kfp
import kfp.components as comp
from kfp.v2 import compiler
from kfp import dsl
import google.cloud.aiplatform as aip
    #image: gcr.io/moth-recognition/simple_component_trial@sha256:35416d2160f76324d4fb3bcc45f53e82707cc13a7cae2dfc689273a84683ca9c

create_step_get_lines = comp.load_component_from_text("""
name: Get Lines
description: Gets the specified number of lines from the input file.

inputs:
- {name: input_1, type: String, description: 'Data for input_1'}
- {name: parameter_1, type: Integer, default: '100', description: 'Number of lines to copy'}

outputs:
- {name: output_1, type: String, description: 'output_1 data.'}

implementation:
  container:
    image: gcr.io/moth-recognition/simple_component_trial:latest
    # command is a list of strings (command-line arguments). 
    # The YAML language has two syntaxes for lists and you can use either of them. 
    # Here we use the "flow syntax" - comma-separated strings inside square brackets.
    command: [
      python3, 
      # Path of the program inside the container
      /pipelines/component/src/program.py,
      --input1-path,
      {inputValue: input_1},
      --param1, 
      {inputValue: parameter_1},
      --output1-path, 
      {outputPath: output_1},
    ]""")

# create_step_get_lines is a "factory function" that accepts the arguments
# for the component's inputs and output paths and returns a pipeline step
# (ContainerOp instance).
#
# To inspect the get_lines_op function in Jupyter Notebook, enter 
# "get_lines_op(" in a cell and press Shift+Tab.
# You can also get help by entering `help(get_lines_op)`, `get_lines_op?`,
# or `get_lines_op??`.

BUCKET_NAME = 'moth-recognition-ml'
BUCKET_NAME = 'moth-recognition-vertex-ai-pipelines-artifacts'
DATASET_VERSION = 'trial'
PIPELINE_ROOT = "gs://{}/pipeline_root/simple_{}".format(BUCKET_NAME, DATASET_VERSION)
# Define your pipeline
@dsl.pipeline(
            pipeline_root=PIPELINE_ROOT,
                name="example-pipeline",
) 
def my_pipeline():
    get_lines_step = create_step_get_lines(
        # Input name "Input 1" is converted to pythonic parameter name "input_1"
        input_1='/pipelines/component/src/tst.txt',
        parameter_1='5',
    #    output_1='/pipelines/component/src/outfile.txt',
     )

# If you run this command on a Jupyter notebook running on Kubeflow,
# you can exclude the host parameter.
# client = kfp.Client()
compiler.Compiler().compile(
   pipeline_func=my_pipeline, package_path='trial.json'
)
#client = kfp.Client()
#print(client)

# Compile, upload, and submit this pipeline for execution.
#client.create_run_from_pipeline_func(my_pipeline, arguments={},
#         mode=kfp.dsl.PipelineExecutionMode.V2_COMPATIBLE)


job = aip.PipelineJob(
     display_name='trial',
     template_path='trial.json',
     pipeline_root=PIPELINE_ROOT,
)

job.run(sync=True)
