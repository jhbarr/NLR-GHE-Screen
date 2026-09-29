Prerequisites
=============

**Software Prerequisites**
----------------------
- You must have `git <https://git-scm.com/>`_ installed on your system in order to clone the repository to your local machine
- To run the program, the repository utilizes the Python package dependency manager `uv <https://docs.astral.sh/uv/>`_

**Installation** 
----------------------------
Once you have all prerequisites installed, clone the repository using 
.. code-block:: bash

    git clone https://github.com/jhbarr/NLR-GHE-Screen

**API Key Setup**
-----------------
- To execute API calls to the OpenTopography database, each user must create a free account and generate a free API key.
- To do so, go to the `OpenTopography website <https://opentopography.org/>`_ and create an account. Once the account is setup, there should be an option to generate a free API key.
- Create a .env file in the newly cloned repository and add the line: API-KEY=users_api_key