#!/usr/bin/env python3
from pathlib import Path

import numpy as np
import polars as pl
from polars.dataframe.frame import DataFrame


def _read_lines(path: Path | str) -> list[str]:
    """Get the lines of text from a file

    Parameters
    ----------
    path : Path | str
        Path to the file

    Returns
    -------
    list[str]
        List with the lines
    """
    with open(path) as f:
        text: str = f.read()

    return str.splitlines(text)

def _step_data_to_table(
    data_step: list[str], nr_part: int, max_b: int, id_heads, bo_heads, timestep
) -> DataFrame:
    ids = np.full(nr_part, np.nan, dtype=np.int32)
    types = np.full(nr_part, np.nan, dtype=np.int32)
    nbs = np.full(nr_part, np.nan, dtype=np.int32)
    idns = np.full((nr_part, max_b), np.nan, dtype=np.int32)
    mols = np.full(nr_part, np.nan, dtype=np.int32)
    bos = np.full((nr_part, max_b), np.nan, dtype=np.float64)
    abos = np.full(nr_part, np.nan, dtype=np.float64)
    nlps = np.full(nr_part, np.nan, dtype=np.int32)
    qs = np.full(nr_part, np.nan, dtype=np.float64)

    for i in range(nr_part):
        line: str = data_step[i]
        t: list[str] = line.split()

        ids[i] = int(t[0])
        types[i] = int(t[1])

        nb: int = int(t[2])
        nbs[i] = nb

        for k in range(nb):
            idns[i, k] = int(t[3 + k])

        mols[i] = int(t[3 + nb])

        for k in range(nb):
            bos[i, k] = float(t[4 + nb + k])

        abos[i] = float(t[4 + 2 * nb])
        nlps[i] = int(float(t[5 + 2 * nb]))
        qs[i] = float(t[6 + 2 * nb])

    idns_dict = {head: idns[:, l] for l, head in enumerate(id_heads)}
    bos_dict = {head: bos[:, l] for l, head in enumerate(bo_heads)}
    step_table = pl.LazyFrame(
        {
            **{
                "timestep": timestep,
                "id": ids,
                "type": types,
                "nb": nbs,
            },
            **idns_dict,
            **{
                "mol": mols,
            },
            **bos_dict,
            **{
                "abo": abos,
                "nlp": nlps,
                "q": qs,
            },
        }
    )

    return step_table.collect()


def _file_to_com_dat(path: Path | str) -> tuple[list[str], list[str]]:
    lines: list[str] = _read_lines(path)

    comments: list[str] = [
        line.removeprefix("#").removesuffix("\n").strip()
        for line in lines
        if line.startswith("#")
    ]
    comments = [com for com in comments if com != ""]
    data: list[str] = [line for line in lines if not line.startswith("#")]

    return comments, data


def _parse_comments(comments: list[str]) -> tuple[list[int], int, int, int]:
    timesteps: list[int] = []
    nr_parts: list[int] = []
    max_bs: list[int] = []

    for com in comments:
        if com.startswith("Timestep"):
            timesteps.append(int(com.split(" ")[1]))
        elif com.startswith("Number of particles"):
            nr_parts.append(int(com.split(" ")[-1]))
        elif com.startswith("Max number of bonds per atom"):
            max_bs.append(int(com.split(" ")[6]))
        elif (
            com == "Particle connection table and bond orders"
            or com == "id type nb id_1...id_nb mol bo_1...bo_nb abo nlp q"
        ):
            continue
        else:
            print("ERROR: Unrecognized comment string")
            quit()

    if len(timesteps) != len(nr_parts) or len(nr_parts) != len(max_bs):
        print("ERROR: Comments parsed unequally")
        quit()

    # Parse number of particles
    if len(np.unique(nr_parts)) == 1:
        nr_part: int = nr_parts[0]
    else:
        print("ERROR: Varying number of particles")
        quit()

    # Parse max number of bonds
    if len(np.unique(max_bs)) == 1:
        max_b: int = max_bs[0]
    else:  # Varying number of max bonds
        max_b = max([mb for mb in np.unique(max_bs)])

    nr_steps: int = len(timesteps)

    return timesteps, nr_part, max_b, nr_steps


def _create_table(
    data: list[str], timesteps: list[int], nr_part: int, max_b: int, nr_steps: int
) -> DataFrame:
    """Create a table with the ReaxFF bond info; contains lots of hardcoded stuff

    Parameters
    ----------
    data : list[str]
        The lines containing the data
    timesteps : list[int]
        The timesteps; for each of which the info is given
    nr_part : int
        Number of particles
    max_b : int
        Max number of bonds
    nr_steps : int
        Number of timesteps

    Returns
    -------
    DataFrame
        Table with the bond info
    """

    id_heads: list[str] = [f"id_{id+1}" for id in range(max_b)]
    bo_heads: list[str] = [f"bo_{id+1}" for id in range(max_b)]
    heads: list[str] = (
        ["timestep", "id", "type", "nb"]
        + id_heads
        + ["mol"]
        + bo_heads
        + ["abo", "nlp", "q"]
    )
    dtypes: list = (
        [pl.Int32] * (4 + max_b + 1)
        + [pl.Float64] * (max_b + 1)
        + [pl.Int32]
        + [pl.Float64]
    )
    table: DataFrame = pl.DataFrame(schema=dict(zip(heads, dtypes)))

    for j in range(nr_steps):
        data_step: list[str] = data[j * nr_part : (j + 1) * nr_part]

        step_table: DataFrame = _step_data_to_table(
            data_step, nr_part, max_b, id_heads, bo_heads, timesteps[j]
        )

        table = table.vstack(step_table)

    # Replace empty values with actual empties
    table = pl.DataFrame(
        [column.replace(-2147483648, None) for column in table]
    ).fill_nan(None)

    return table


def file_to_ReaxFF_bond_table(
    file_path: Path | str,
    sort: str | list[str] = ["timestep", "id"],
    save=False,
    save_path: Path | str = "",
    delete_file=False,
) -> DataFrame:
    """Turns a LAMMPS ReaxFF bond file into a `polars` table and optionally save it as CSV file

    Because the standard output format of `fix 1 all reaxff/bonds ...` can contain rows of varying width,
    the file normally can't easily be parsed. This function solves that problem.
    While suboptimal for memory usage, the timestep is added to each row for simplicity of use.

    Parameters
    ----------
    file_path : Path | str
        Path to the bond file
    sort : str | list[str], optional
        The parameters columns by which to sort the table at the end, by default ["timestep", "id"]
    save : bool, optional
        Whether to save the table, by default False
    save_path : Path | str, optional
        The path (including name and extension) to save the table to if `save` is True, by default ""
    delete_file : bool, optional
        Whether to delete the original bond file, this could save space, by default False

    Returns
    -------
    DataFrame
        A table containing all the bond information
    """
    comments, data = _file_to_com_dat(file_path)
    timesteps, nr_part, max_b, nr_steps = _parse_comments(comments)
    table: DataFrame = _create_table(data, timesteps, nr_part, max_b, nr_steps)

    if isinstance(sort, str):
        sort = [sort]

    table = table.sort(["timestep"] + sort)

    if save:
        table.write_csv(save_path)
    if delete_file:
        Path(file_path).unlink()

    return table

