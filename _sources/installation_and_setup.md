# Installation and setup

1. Clone the catlas repo 
    ```
    git clone https://github.com/ulissigroup/catlas.git
    OR
    git clone git@github.com:ulissigroup/catlas.git
    ```
2. Install catlas locally
    ```
    cd ~/catlas && python setup.py develop
    ```
3. Add [model checkpoints](https://github.com/Open-Catalyst-Project/ocp/blob/main/MODELS.md) to a folder in the base directory called `ocp_checkpoints`

## Adding parity data
The parity data files have been omitted from the repo because they are large. We will have them as downloadable links. If you need them now. Please email bwander@andrew.cmu.edu. They should be put in `catlas/catlas/parity/df_pkls/` and should be named to match the model checkpoint name.

## Advanced options
If you plan to use custom model architectures, they must be compatible with the OCP infra. This is currently supported. To do this you have to
1. clone the [ocp repo](https://github.com/Open-Catalyst-Project/ocp)
2. install it locally `cd ~/ocp && python setup.py develop`
3. make your changes & train your model. Then just use the checkpoint as you would for any other catlas run
