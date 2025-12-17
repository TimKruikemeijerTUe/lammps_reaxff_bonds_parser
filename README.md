# LAMMPS_ReaxFF_bonds_parser
Turns a LAMMPS ReaxFF bond file into a `polars` table and optionally save it as CSV file

## Usage
```python
from lmp_reaxff_bonds_reader import file_to_ReaxFF_bond_table

ReaxFF_bonds = file_to_ReaxFF_bond_table("bonds.txt", save=True, save_path="bonds.csv")
```

## Dependencies
This script relies on [numpy](https://numpy.org/) and [polars](https://pola.rs/)
