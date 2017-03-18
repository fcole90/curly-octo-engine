curly_octo_engine
-----------------

This project aims to create a movie recommendation system trough a neural network
that considers visual, audio and text data to provide ratings estimates.

## Current best result
| data              | value                 |
|-------------------|-----------------------|
| batch             | 200                   |
| learning rate     | 0.01                  |
| optimizer         | Gradient Descent      |
| loss function     | differentiable RMSE   |
| diff_argmax power | 100                   |
| layers            | 1                     |
| **RMSE**          | **1.0572**            |
| **Acc s/ns**      | **83.58%**            |
| **Accuracy**      | **35.26%**            |

## Reference results
| Project                   | RMSE   |
|---------------------------|--------|
| curly-octo-engine         | 1.0572 |
| Netflix's Cinematch       | 0.9525 |
| Netflix Prize Target      | 0.8572 |
| BellKor's Pragmatic Chaos | 0.8558 |
| The Ensemble              | 0.8554 |

*Note: RMSE is probably not the best metric to estimate the accuracy over the
 predictions. Tough in this field it's a widely used one and allows to make
 comparisons with other approaches.*


## Requirements
*Follow the [tensorflow guide](https://www.tensorflow.org/install/install_linux)
 to create a virtualenv with the appropriate Python packages.*
  
System: python3, python3-virtualenv.

Python packages: tensorflow, numpy, sklearn.

*Other packages may be required in order to run the dataset collecting parts.*

## Project Structure
| folder                    | content               |
|---------------------------|-----------------------|
| image_dataset             | scripts to download images and create a dataset |
| soundtrack_dataset        | scripts to collect Spotify music data and create a dataset |
| textual_dataset           | scripts to collect IMDB movie keywords and movie plots and create a dataset |
| visual_data_simulation    | experiment to test the behaviour using only visual data |
| tools                     | helper functions library |
| math_fun                  | some experimental fun |

## First tests
Run ```experiment_alpha.py```. You'll get updates on the
experiment estimators. To tweak the settings, change the values of the 
settings at the top of the file. *Yes, I should split the settings from 
the source, it's on the todo list.*
 
## Output
The output shows the current values in the left column and the best value of the experiment on the second one.
The estimates are rate accuracy, suggestion accuracy and RMSE:
 - the rate accuracy ```Acc``` is the percentage of correct rate predictions;
 - the suggestion accuracy ```Acc s/ns``` is the percentage of correct suggestion
   predictions, a movie is suggested if it has a rate greater or equal to three;
 - the root mean squared error, the metric used by the Netflix Prize too.
The estimates are made over the validation set to avoid overfitting.
A ```GOOD``` on the right highlights an improvement on the best RMSE.
 