To run the virtual environment:

 ```env\Scripts\activate```

In order to add the required libraries inside the env:

```pip install numpy```

## Requirements

To run this project, you need to have Python 3.x installed. Then, install the following Python libraries:

- Flask:     pip install flask
- Pandas:    pip install pandas
- NumPy:     pip install numpy
- scikit-learn: pip install scikit-learn
- Invoke:    pip install invoke
- tqdm: pip install tqdm

These libraries provide the necessary environment for building the REST API, handling and processing data, training models, and running project tasks.



## Running each task

To prepare the csv file
`invoke data`

To train model

`invoke train`

To run api

`invoke api`


