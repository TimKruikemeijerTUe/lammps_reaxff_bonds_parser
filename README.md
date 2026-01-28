# lammps_reaxff_bonds_parser
Turns a LAMMPS ReaxFF bond file into a `polars` table and optionally save it as CSV file. Can be used with large files.

## Usage
```python
from lammps_reaxff_bonds_parser import file_to_ReaxFF_bond_table

ReaxFF_bonds = file_to_ReaxFF_bond_table("bonds.txt", large_file=False, save=True, save_path="bonds.csv")
```

## Installation
Local install using pip:
```
pip install git+https://github.com/TimKruikemeijerTUe/LAMMPS_ReaxFF_bonds_parser.git
```
or with `tqdm` 
```
pip install "lammps_reaxff_bonds_parser[tqdm] @  git+https://github.com/TimKruikemeijerTUe/LAMMPS_ReaxFF_bonds_parser.git
"```

## Dependencies
This script relies on [numpy](https://numpy.org/) and [polars](https://pola.rs/). [tqdm](https://tqdm.github.io/) is optionally used for large file progress tracking.
