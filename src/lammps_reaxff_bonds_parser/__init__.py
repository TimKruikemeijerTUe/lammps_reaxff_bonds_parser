#!/usr/bin/env python3
import sys
from importlib.util import find_spec
from itertools import islice
from pathlib import Path

import numpy as np
import polars as pl
from polars.dataframe.frame import DataFrame
from polars.lazyframe.frame import LazyFrame

tqdm_installed: bool = find_spec("tqdm") is not None
if tqdm_installed:
    from tqdm import tqdm


def _step_data_to_table(
    data_step: list[str],
    nr_part: int,
    max_b: int,
    id_heads: list[str],
    bo_heads: list[str],
    timestep: int,
) -> DataFrame:
    """Create step table from the step data lines. Involves many hardcoded info.

    Parameters
    ----------
    data_step : list[str]
        The data lines
    nr_part : int
        Number of particles
    max_b : int
        Maximum number of neighbours
    id_heads : list[str]
        List of the columns names of the neighbouring IDs
    bo_heads : list[str]
        List of the columns names of the neighbouring BOs
    timestep : int
        The timestep of the this data

    Returns
    -------
    DataFrame
        The step table
    """
    # Sentinel value to replace empty values later. Largest signed 32 bit int
    sentinel = -2147483648

    # Initialize the arrays for easier access
    ids = np.full(nr_part, sentinel, dtype=np.int32)
    types = np.full(nr_part, sentinel, dtype=np.int32)
    nbs = np.full(nr_part, sentinel, dtype=np.int32)
    idns = np.full((nr_part, max_b), sentinel, dtype=np.int32)
    mols = np.full(nr_part, sentinel, dtype=np.int32)
    bos = np.full((nr_part, max_b), sentinel, dtype=np.float64)
    abos = np.full(nr_part, sentinel, dtype=np.float64)
    nlps = np.full(nr_part, sentinel, dtype=np.int32)
    qs = np.full(nr_part, sentinel, dtype=np.float64)

    # Iterate over the lines
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

    # Dynamically create dicts
    idns_dict = {head: idns[:, id] for id, head in enumerate(id_heads)}
    bos_dict = {head: bos[:, id] for id, head in enumerate(bo_heads)}
    # Create frame from arrays
    step_table = pl.DataFrame(
        {
            "timestep": timestep,
            "id": ids,
            "type": types,
            "nb": nbs,
            **idns_dict,
            "mol": mols,
            **bos_dict,
            "abo": abos,
            "nlp": nlps,
            "q": qs,
        },
    )

    return step_table


def _file_to_com(path: Path | str) -> list[str]:
    r"""Get all (non-empty) lines with '#' prefix. Removes '#' or '\n'.

    Parameters
    ----------
    path : Path | str
        Path to the text file

    Returns
    -------
    list[str]
        List with comment lines
    """
    with Path(path).open() as f:
        comments: list[str] = [
            line.removeprefix("#").removesuffix("\n").strip()
            for line in f
            if line.startswith("#")
        ]

    comments = [com for com in comments if com != ""]

    return comments


def _file_to_data(path: Path | str) -> list[str]:
    r"""Get all lines without '#' prefix. Removes '\n'.

    Parameters
    ----------
    path : Path | str
        Path to the text file

    Returns
    -------
    list[str]
        List with data lines
    """
    with Path(path).open() as f:
        data: list[str] = [
            line.removesuffix("\n").strip() for line in f if not line.startswith("#")
        ]

    return data


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
        elif com in {
            "Particle connection table and bond orders",
            "id type nb id_1...id_nb mol bo_1...bo_nb abo nlp q",
        }:
            continue
        else:
            print("ERROR: Unrecognized comment string")
            sys.exit()

    if len(timesteps) != len(nr_parts) or len(nr_parts) != len(max_bs):
        print("ERROR: Comments parsed unequally")
        sys.exit()

    # Parse number of particles
    if len(np.unique(nr_parts)) == 1:
        nr_part: int = nr_parts[0]
    else:
        print("ERROR: Varying number of particles")
        sys.exit()

    # Parse max number of bonds
    if len(np.unique(max_bs)) == 1:
        max_b: int = max_bs[0]
    else:  # Varying number of max bonds
        max_b = max(list(np.unique(max_bs)))

    nr_steps: int = len(timesteps)

    return timesteps, nr_part, max_b, nr_steps


def _create_table(
    data: list[str],
    timesteps: list[int],
    nr_part: int,
    max_b: int,
    nr_steps: int,
) -> DataFrame:
    """Create a table with the ReaxFF bond info; contains lots of hardcoded stuff.

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
    id_heads: list[str] = [f"id_{id + 1}" for id in range(max_b)]
    bo_heads: list[str] = [f"bo_{id + 1}" for id in range(max_b)]
    heads: list[str] = [
        "timestep",
        "id",
        "type",
        "nb",
        *id_heads,
        "mol",
        *bo_heads,
        "abo",
        "nlp",
        "q",
    ]
    dtypes: list = (
        [pl.Int32] * (4 + max_b + 1)
        + [pl.Float64] * (max_b + 1)
        + [pl.Int32]
        + [pl.Float64]
    )
    table: DataFrame = pl.DataFrame(schema=dict(zip(heads, dtypes, strict=True)))

    for j in range(nr_steps):
        data_step: list[str] = data[j * nr_part : (j + 1) * nr_part]

        step_table: DataFrame = _step_data_to_table(
            data_step,
            nr_part,
            max_b,
            id_heads,
            bo_heads,
            timesteps[j],
        )

        table = table.vstack(step_table)

    # Replace empty values with actual empties
    sentinel = -2147483648  # Largest signed 32 bit int
    table = pl.DataFrame([column.replace(sentinel, None) for column in table]).fill_nan(
        None,
    )

    return table


def _create_table_seq(
    file_path: Path | str,
    timesteps: list[int],
    nr_part: int,
    max_b: int,
    nr_steps: int,
    sort: list[str],
    save_path: Path | str,
) -> LazyFrame:
    """Create a table with the ReaxFF bond info. Contains lots of hardcoded stuff. Step by step for memory reasons.

    Parameters
    ----------
    file_path : Path | str
        Path to the ReaxFF bonds file
    timesteps : list[int]
        The timesteps of the file
    nr_part : int
        Particles per timestep
    max_b : int
        Max number of bonds
    nr_steps : int
        Number of timesteps
    sort : list[str]
        Headers by which to sort
    save_path : Path | str
        The path (including name and extension) to save the table to

    Returns
    -------
    LazyFrame
        Table with the bond info
    """
    id_heads: list[str] = [f"id_{id + 1}" for id in range(max_b)]
    bo_heads: list[str] = [f"bo_{id + 1}" for id in range(max_b)]

    sentinel = -2147483648  # Largest signed 32 bit int

    first_table = True  # For header writing
    Path(save_path).open(mode="w", encoding="utf8").close()  # Empty file

    # Sequential for memory
    if tqdm_installed:
        pbar = tqdm(total=nr_steps, desc="ReaxFF written: ")

    stepper = iter(timesteps)
    with (
        open(file_path) as text_file,
        open(save_path, mode="a", encoding="utf8") as f,
    ):
        for line in text_file:
            if not line.startswith("#"):
                data = [line, *list(islice(text_file, 0, nr_part - 1))]

                table = _step_data_to_table(
                    data,
                    nr_part,
                    max_b,
                    id_heads,
                    bo_heads,
                    next(stepper),
                )

                table = table.sort(sort)
                table = pl.DataFrame(
                    [column.replace(sentinel, None) for column in table],
                ).fill_nan(None)

                if first_table:
                    table.write_csv(f)
                    first_table = False
                else:
                    table.write_csv(f, include_header=False)

                if tqdm_installed:
                    pbar.update()

    if tqdm_installed:
        pbar.close()

    ids_schema = {f"id_{id + 1}": pl.Int32 for id in range(max_b)}

    return pl.scan_csv(save_path, schema_overrides=ids_schema)


def file_to_ReaxFF_bond_table(
    file_path: Path | str,
    sort: str | list[str] | None = None,
    *,
    large_file: bool = False,
    save: bool = False,
    save_path: Path | str = "",
    delete_file: bool = False,
) -> LazyFrame:
    """Turn a LAMMPS ReaxFF bond file into a `polars` table and optionally save it as CSV file.

    Because the standard output format of `fix 1 all reaxff/bonds ...` can contain rows of varying width,
    the file normally can't easily be parsed. This function solves that problem.
    While suboptimal for memory usage, the timestep is added to each row for simplicity of use.

    Parameters
    ----------
    file_path : Path | str
        Path to the bond file
    sort : str | list[str] | None, optional
        The parameters columns by which to sort the table at the end, by default ["timestep", "id"]
    large_file : bool, optional
        Whether to optimize space usage, by default False
    save : bool, optional
        Whether to save the table. Needed for `large_file`, by default False
    save_path : Path | str, optional
        The path (including name and extension) to save the table to if `save` is True, by default ""
    delete_file : bool, optional
        Whether to delete the original bond file, this could save space, by default False

    Returns
    -------
    LazyFrame
        A table containing all the bond information
    """
    if sort is None:
        sort = ["timestep", "id"]
    elif isinstance(sort, str):
        sort = [sort]

    comments = _file_to_com(file_path)
    timesteps, nr_part, max_b, nr_steps = _parse_comments(comments)

    if large_file:
        if not save:
            print("ERROR: For large files saving is needed for useful behaviour")
            sys.exit()

        del comments  # For space

        table: LazyFrame = _create_table_seq(
            file_path,
            timesteps,
            nr_part,
            max_b,
            nr_steps,
            sort,
            save_path,
        )

    else:
        data: list[str] = _file_to_data(file_path)

        table_d: DataFrame = _create_table(data, timesteps, nr_part, max_b, nr_steps)
        table_d = table_d.sort(sort)

        if save:
            table_d.write_csv(save_path)

        table = table_d.lazy()

    if delete_file:
        Path(file_path).unlink()

    return table
