# hepML

A package for machine learning studies with HEP

When installing make sure to initialise and update the submodule:

  - ```git submodule init```

  - ```git submodule update```

Start by turning your root trees into numpy arrays! Then carry out some python based data analysis.

For convenience, if you have access to cvmfs and don't want to use tensorflow or keras just use:

  - ```source lcgenv.sh```

Or, for a more complete experience, setup with anaconda:

 - Install anaconda v4 (I couldn't get it to work with v5) e.g. using
 
    - ```bash /nfs/dust/cms/user/elwoodad/Anaconda2-4.4.0-Linux-x86_64.sh```

    - choose whether you want anaconda added to bashrc, overwriting system default packages (i didn't)

    - point the anacondaSetup.sh file to your install (see mine as example)

 - Make a conda environment with all the software you'll need (hepML)

   - When running on the naf you can do:

    - ```conda env create -f environment.yml```

   - Or you can do a generic install of all the relevant tools:
   
    - ```conda create -n hepML -c nlesc root root_numpy keras pandas seaborn scikit-learn tensorflow tensorflow-gpu pydot```
    - NOTE: I've had problems on some systems getting root to install, so you can optionally leave out root and root_numpy if there is an error and install them separately

   - root and numpy are installed using instructions in https://nlesc.gitbooks.io/cern-root-conda-recipes/content/index.html , but the environment file should do everything

 - Activate your environment and you're good to go

   - ```source activate hepML```




Set a basic workflow in run.py. Test it with:

  - ```python run.py``` 


