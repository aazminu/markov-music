# Data Directory

This folder holds large third-party datasets that power the `markov-music` 
experiments. Only lightweight metadata, helper scripts, and Git submodule 
pointers are checked into version control. The actual data either lives in its 
own upstream repository or is downloaded locally on demand.

## Nottingham Dataset (GitHub submodule)
- **Path:** `data/nottingham_github`
- **Source:** https://github.com/jukedeck/nottingham-dataset
- **How to fetch:** `git submodule update --init --recursive data/nottingham_github`
- **License:** GNU GPLv3. Redistribution or modification must follow the copyleft
terms published with the dataset.

```1:20:data/nottingham_github/LICENSE.md
GNU GENERAL PUBLIC LICENSE
Version 3, 29 June 2007
```

## Bach Chorales (Kaggle)
- **Path:** `data/bach_chorales`
- **Source:** https://www.kaggle.com/datasets/pranjalsriv/bach-chorales-2
- **How to fetch:** Run `./scripts/fetch_bach_chorales.sh` after installing the 
Kaggle CLI (`pip install kaggle`), creating `~/.kaggle/kaggle.json`, and 
ensuring `unzip` is available.
- **License:** See the Kaggle dataset page for the latest license and 
redistribution terms. Respect any attribution, non-commercial, or share-alike 
requirements listed there.

## POP909 Dataset (GitHub submodule)
- **Path:** `data/pop909`
- **Source:** https://github.com/music-x-lab/POP909-Dataset
- **How to fetch:** `git submodule update --init --recursive data/pop909`
- **License:** MIT License; retain the copyright notice when redistributing.

```1:13:data/pop909/LICENSE
MIT License

Copyright (c) 2020 Music X Lab
```

## Version-control guidelines
- The `.gitmodules` file pins the submodule revisions for Nottingham (GitHub) 
and POP909. Update them with `git submodule update --remote <path>` and commit 
the new pointers when you need newer data.
- `.gitignore` excludes `data/bach_chorales/` and `data/nottingham_ifdo/` so 
local downloads do not end up in commits. If you add more on-demand datasets, 
list their directories there as well.
- When sharing the repo, document any steps you took to preprocess the data so 
collaborators can reproduce them locally.

