.. _stpt_contributing:

Contributing to development
===========================

The code is kept in a git repository at https://gitlab.ast.cam.ac.uk/imaxt/stpt-pipeline

Setting up git
--------------

The first thing is to install git if it is not already installed. Follow the
instructions here_.

.. _here: https://git-scm.com/book/en/v2/Getting-Started-Installing-Git


Then set up your user name and email address.

.. code-block:: bash

   $ git config --global user.name "John Doe"
   $ git config --global user.email johndoe@example.com

You can also configure the editor that will be used when some text is needed
(e.g. to edit commit messages):

.. code-block:: bash

   $ git config --global core.editor emacs

Note that this needs to be done only once.


Setting up your Gitlab account
------------------------------

Once registered, login into your gitlab account and
`add your SSH key <https://gitlab.ast.cam.ac.uk/profile/keys>`_ to your
account in order to be able to checkout and commit code.


Contributing code
-----------------

1. Clone the repository you want to work on locally::

     $ git clone git@gitlab.ast.cam.ac.uk:imaxt/stpt-pipeline.git

2. Install your local copy either in a virtualenv or a conda environment. Assuming you have
   virtualenvwrapper installed, this is how you setup your local development
   environment::

     $ mkvirtualenv imaxt
     $ cd imc-pipeline
     $ pip install -r requirements_dev.txt
     $ python setup.py develop

3. In order to contribute code, first create a branch where your changes are
   going to be done::

     $ git checkout -b name-of-your-bugfix-or-feature
     $ git push --set-upstream origin name-of-your-bugfix-or-feature

   Now you can make your changes. Note that the code should adhere to the
   :ref:`Python style guide<imaxt:style-guide-main>` and
   :ref:`Documenting Python APIs<imaxt:numpydoc-format>` documents.

4. When all changes are done, check that they pass flakes8 and the tests::

     $ python setup.py flake8
     $ python setup.py test

   It is useful as well to have autopep8 and isort run on the code.
   Check also that the documents, located in the ``doc`` directory build ok::

     $ python setup.py build_sphinx

The command above will build the documentation in ``build/sphinx/html``.

5. Commit your changes and push your branch::

     $ git add .
     $ git commit -m "Your detailed description of your changes."
     $ git push

6. Submit a merge request `<https://gitlab.ast.cam.ac.uk/imaxt/stpt-pipeline/merge_requests/new>`_
   through the website.

7. If this contribution is associated to a Jira or Gitlab ticket, update the ticket with
   the URL of the merge request (and possibly change its status to "To Review").

Running the pipeline in development
-----------------------------------

In order to run the pipeline in development mode you will need to install the Owl Client::

     $ pip install owl-client

Given a pipeline definition file (see :ref:`stpt_pipedef`), the pipeline can then be run using::

     env LOGLEVEL=DEBUG owl execute /path/to/config.yaml

