# hepML

A package for machine learning studies with HEP

Start by turning your root trees into numpy arrays!

To do this install root_numpy... http://scikit-hep.org/root_numpy/start.html

For convenience, if you have access to cvmfs and don't want to use tensorflow or keras just use:

  - ```source lcgenv.sh```

Or, for a more complete experience, setup with anaconda:

 - Install anaconda v4 (I couldn't get it to work with v5) e.g. using /nfs/dust/cms/user/elwoodad/Anaconda2-4.4.0-Linux-x86_64.sh

    - choose whether you want anaconda added to bashrc, overwriting system default packages (i didn't)

    - point the anacondaSetup.sh file to your install (see mine as example)

 - Make a conda environment with all the software you'll need (hepML)

   - ```conda env create -f environment.yml```

   - root and numpy are installed using instructions in https://nlesc.gitbooks.io/cern-root-conda-recipes/content/index.html , but the environment file shoudl do everything

 - Activate your environment and you're good to go

   - ```source activate hepML```




Set a basic workflow in run.py. Test it with:

  - ```python run.py``` 


