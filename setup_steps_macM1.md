These are more-or-less the steps followed to get scripts working in a virtual environment on a MacBook Pro (M1 chip). 


1) open iterm with rosetta (follow these steps: https://www.courier.com/blog/tips-and-tricks-to-setup-your-apple-m1-for-development/)
2) install pip3 or pip (if not already available)
3) make virtual environment using python3.9 (note: some google libraries don't seem to work with python10 yet)
  - install virtualenv: https://sourabhbajaj.com/mac-setup/Python/virtualenv.html or https://gist.github.com/pandafulmanda/730a9355e088a9970b18275cb9eadef3
  - brew install python@3.9
  - virtualenv -p python3.9 trial
  - source [venv]/bin/activate
4) install the following libraries:
  - pip install google-cloud-automl
  - pip install --upgrade google-cloud-aiplatform
  - pip install numpy
  - pip install Pillow
  - pip install pandas
  - pip install sqlalchemy
  - pip install psycopg2-binary
  - pip install pyarrow
  - pip install gcsfs
  - pip install -U google-cloud-pipeline-components
  - pip install google-cloud-secret-manager
  - pip install --upgrade google-cloud-datastore
  - pip install matplotlib
  - pip install jinja2
  - pip install opencv-python
  - pip install imutils
  - pip install scikit-image
  - python3 -m pip install tensorflow-macos
