"""Workflows."""
from __future__ import annotations

from collections.abc import MutableMapping, Sequence
from dataclasses import dataclass
from inspect import signature
from os import PathLike, path
from pathlib import Path
from tempfile import TemporaryDirectory
import textwrap
from typing import Mapping, Optional, Union

from morfeus.conformer import ConformerEnsemble
import numpy as np
import os
import subprocess

from polanyi import config
from polanyi.geometry import two_frags_from_bo
from polanyi.interpolation import interpolate_geodesic
from polanyi.pyscf import OptResults, ts_from_gfnff_python, ts_from_gfnff, ts_from_gfnff_ci_python, ts_from_gfnff_ci
from polanyi.typing import Array1D, Array2D, ArrayLike2D
from polanyi.xtb import (
    opt_crest,
    opt_xtb,
    parse_energy,
    run_xtb,
    wbo_xtb,
    XTBCalculator,
)


@dataclass
class Results:
    """Results of TS optimization."""

    opt_results: OptResults
    coordinates_opt: Array2D
    shift_results: Optional[ShiftResults] = None


@dataclass
class ShiftResults:
    """Results of energy shift calculation."""

    energy_shift: float
    energy_diff_gfn: float
    energy_diff_ff: float
    energies_gfn: list[float]
    energies_ff: list[float]


def opt_ts_python(
    elements: Union[Sequence[int], Sequence[str]],
    coordinates: Sequence[Array2D],
    coordinates_guess: Optional[Array2D] = None,
    e_shift: Optional[float] = None,
    kw_calculators: Optional[Mapping] = None,
    kw_shift: Optional[Mapping] = None,
    kw_opt: Optional[Mapping] = None,
    kw_interpolation: Optional[Mapping] = None,
) -> Results:
    """Optimize transition state with xtb-python and PySCF.
    Args:
        elements: elements as symbols or numbers
        coordinates: sequence containing the coordinates of each ground states [Å]
        coordinates_guess: initial guess for the transition state [Å]
        e_shift: energy shift between reference (GFN2-xTB by default) and GFN-FF reaction energies
        kw_calculators: parameters for topologies calculation
        kw_shift: parameters for energy shift calculation
        kw_opt: parameters for optimization
        kw_interpolation: parameters for the TS interpolation
    Returns:
        results: results of the TS optimization
    """
    if kw_opt is None:
        kw_opt = {}
    if kw_shift is None:
        kw_shift = {}
    if kw_calculators is None:
        kw_calculators = {}
    if kw_interpolation is None:
        kw_interpolation = {}
    
    calculators = setup_gfnff_calculators_python(
        elements, coordinates, **kw_calculators
    )

    shift_results: Optional[ShiftResults]
    if e_shift is None:
        shift_results = calculate_e_shift_xtb_python(calculators, **kw_shift)
        e_shift = shift_results.energy_shift
    else:
        shift_results = None
    
    if coordinates_guess is None:
        n_images = kw_interpolation.get("n_images")
        if n_images is None:
            n_images = signature(interpolate_geodesic).parameters["n_images"].default
        path = interpolate_geodesic(elements, coordinates, **kw_interpolation)
        coordinates_guess = path[n_images // 2]
    
    opt_results = ts_from_gfnff_python(
        elements, coordinates_guess, calculators, e_shift=e_shift, **kw_opt
    )

    results = Results(
        opt_results=opt_results,
        coordinates_opt=opt_results.coordinates[-1],
        shift_results=shift_results,
    )

    return results


def opt_ts_ci_python(
    elements: Union[Sequence[int], Sequence[str]],
    coordinates: Sequence[Array2D],
    coordinates_guess: Optional[Array2D] = None,
    e_shift: Optional[float] = None,
    kw_calculators: Optional[Mapping] = None,
    kw_shift: Optional[Mapping] = None,
    kw_opt: Optional[Mapping] = None,
    kw_interpolation: Optional[Mapping] = None,
) -> Array2D:
    """Optimize transition state with xtb-python and PySCF using conical intersection.
    Args:
        elements: elements as symbols or numbers
        coordinates: sequence containing the coordinates of each ground states [Å]
        coordinates_guess: initial guess for the transition state [Å]
        e_shift: energy shift between reference (GFN2-xTB by default) and GFN-FF reaction energies
        kw_calculators: parameters for topologies calculation
        kw_shift: parameters for energy shift calculation
        kw_opt: parameters for optimization
        kw_interpolation: parameters for the TS interpolation
    Returns:
        coordinates_op: coordinates of the optimised transition state [Å]
    """
    if kw_opt is None:
        kw_opt = {}
    if kw_shift is None:
        kw_shift = {}
    if kw_calculators is None:
        kw_calculators = {}
    if kw_interpolation is None:
        kw_interpolation = {}
    
    calculators = setup_gfnff_calculators_python(
        elements, coordinates, **kw_calculators
    )

    shift_results: Optional[ShiftResults]
    if e_shift is None:
        shift_results = calculate_e_shift_xtb_python(calculators, **kw_shift)
        e_shift = shift_results.energy_shift
    else:
        shift_results = None
    
    if coordinates_guess is None:
        n_images = kw_interpolation.get("n_images")
        if n_images is None:
            n_images = signature(interpolate_geodesic).parameters["n_images"].default
        path = interpolate_geodesic(elements, coordinates, **kw_interpolation)
        coordinates_guess = path[n_images // 2]
    
    coordinates_opt = ts_from_gfnff_ci_python(elements, coordinates_guess, calculators, e_shift=e_shift, **kw_opt)

    return coordinates_opt


def opt_ts_ci(
    elements: Union[Sequence[int], Sequence[str]],
    coordinates: Sequence[Array2D],
    coordinates_guess: Optional[Array2D] = None,
    atomic_charges: Optional[list[float]] = None,
    e_shift: Optional[float] = None,
    e_diff_ref: Optional[float] = None,
    kw_calculators: Optional[Mapping] = None,
    kw_shift: Optional[Mapping] = None,
    kw_opt: Optional[Mapping] = None,
    kw_interpolation: Optional[Mapping] = None,
) -> Results:
    """Optimize transition state with xtb command line and PySCF using conical intersection.
    Args:
        elements: elements as symbols or numbers
        coordinates: sequence containing the coordinates of each ground states [Å]
        coordinates_guess: initial guess for the transition state [Å]
        atomic_charges: atomic charges (not implemented yet)
        e_shift: energy shift between reference (GFN2-xTB by default) and GFN-FF reaction energies
        e_diff_ref: reference reaction energy [Eh]. If provided, it is used instead of GFN2-xTB in the energy shift calculation
        kw_calculators: parameters for topologies calculation
        kw_shift: parameters for energy shift calculation
        kw_opt: parameters for optimization
        kw_interpolation: parameters for the TS interpolation
    Returns:
        results: results of the TS optimization
    """
    if kw_opt is None:
        kw_opt = {}
    if kw_shift is None:
        kw_shift = {}
    if kw_calculators is None:
        kw_calculators = {}
    if kw_interpolation is None:
        kw_interpolation = {}
    
    topologies = setup_gfnff_calculators(elements, coordinates, atomic_charges=atomic_charges, **kw_calculators)
    
    shift_results: Optional[tuple[float, float, float]]
    if e_shift is None:
        shift_results = calculate_e_shift_xtb(elements, coordinates, topologies, e_diff_ref=e_diff_ref, **kw_shift)
        e_shift = shift_results[0]
    else:
        shift_results = None
    
    if coordinates_guess is None:
        n_images = kw_interpolation.get("n_images")
        if n_images is None:
            n_images = signature(interpolate_geodesic).parameters["n_images"].default
        path = interpolate_geodesic(elements, coordinates, **kw_interpolation)
        coordinates_guess = path[n_images // 2]
    
    coordinates_opt = ts_from_gfnff_ci(elements, coordinates_guess, topologies, e_shift=e_shift, **kw_opt)

    return coordinates_opt

def opt_ts(
    elements: Union[Sequence[int], Sequence[str]],
    coordinates: Sequence[Array2D],
    coordinates_guess: Optional[Array2D] = None,
    atomic_charges: Optional[list[float]] = None,
    e_shift: Optional[float] = None,
    kw_calculators: Optional[Mapping] = None,
    kw_shift: Optional[Mapping] = None,
    kw_opt: Optional[Mapping] = None,
    kw_interpolation: Optional[Mapping] = None,
) -> Results:
    """Optimize transition state with xtb command line and PySCF.
    Args:
        elements: TS elements as symbols or numbers
        coordinates: sequence containing the coordinates of each ground states [Å]
        coordinates_guess: initial guess for the transition state [Å]
        atomic_charges: atomic charges (not implemented yet)
        e_shift: energy shift between reference (GFN2-xTB by default) and GFN-FF reaction energies
        kw_calculators: parameters for topologies calculation
        kw_shift: parameters for energy shift calculation
        kw_opt: parameters for optimization
        kw_interpolation: parameters for the TS interpolation
    Returns:
        results: Results of the TS optimization
    """
    if kw_opt is None:
        kw_opt = {}
    if kw_shift is None:
        kw_shift = {}
    if kw_calculators is None:
        kw_calculators = {}
    if kw_interpolation is None:
        kw_interpolation = {}
    topologies = setup_gfnff_calculators(elements, coordinates, atomic_charges=atomic_charges, **kw_calculators)
    shift_results: Optional[tuple[float, float, float]]
    if e_shift is None:
        shift_results = calculate_e_shift_xtb(elements, coordinates, topologies, **kw_shift)
        e_shift = shift_results[0]
    else:
        shift_results = None
    if coordinates_guess is None:
        n_images = kw_interpolation.get("n_images")
        if n_images is None:
            n_images = signature(interpolate_geodesic).parameters["n_images"].default
        path = interpolate_geodesic(elements, coordinates, **kw_interpolation)
        coordinates_guess = path[n_images // 2]
    
    opt_results = ts_from_gfnff(elements, coordinates_guess, topologies, e_shift=e_shift, **kw_opt)

    # Save the optimisation steps if path for optimisation is given
    if "path" in kw_opt and kw_opt["path"] is not None:
        run_path = kw_opt["path"]
        os.makedirs(run_path, exist_ok=True)
        temp_xyz = f"{run_path}/opt_steps.xyz"
        output_pdb = f"{run_path}/opt_steps.pdb"
        with open(temp_xyz, "w") as f:
            f.write(f"{len(elements)}\n")
            f.write("initial guess\n")
            for element, coord_elem in zip(elements, coordinates_guess):
                f.write(f"{element} {coord_elem[0]} {coord_elem[1]} {coord_elem[2]}\n")
            for i, coord in enumerate(opt_results.coordinates):
                f.write(f"{len(elements)}\n")
                f.write(f"step {i}\n")
                for element, coord_elem in zip(elements, coord):
                    f.write(f"{element} {coord_elem[0]} {coord_elem[1]} {coord_elem[2]}\n")
                f.write("\n")
        # Convert the xyz file to pdb for trajectory visualization
        cmd_openbabel = f"obabel -ixyz {temp_xyz} -O {output_pdb}"
        subprocess.run(cmd_openbabel.split())
        subprocess.run(f"rm {temp_xyz}".split())
    
    results = Results(
        opt_results=opt_results,
        coordinates_opt=opt_results.coordinates[-1],
        shift_results=shift_results,
    )

    return results


def setup_gfnff_calculators(
    elements: Union[Sequence[int], Sequence[str]],
    coordinates: Sequence[ArrayLike2D],
    atomic_charges: Optional[list[float]] = None,
    keywords: Optional[list[str]] = None,
    xcontrol_keywords: Optional[MutableMapping[str, list[str]]] = None,
    paths: Optional[Sequence[Union[str, PathLike]]] = None,
) -> list[bytes]:
    """Sets up force fields for GFNFF calculation."""

    # Set the xtb keywords for the GFN-FF calculations
    if keywords is None:
        keywords = []
    keywords = set([keyword.strip().lower() for keyword in keywords])
    # Give the --gfnff keyword (--gfn2 by default)
    keywords.add("--gfnff")

    if paths is None:
        temp_dirs = [
            TemporaryDirectory(dir=config.TMP_DIR) for i in range(len(coordinates))
        ]
        xtb_paths = [Path(temp_dir.name) for temp_dir in temp_dirs]
    else:
        xtb_paths = [Path(path) for path in paths]

    topologies = []
    for coordinates_, xtb_path in zip(coordinates, xtb_paths):
        if atomic_charges and not path.isfile(xtb_path / "charges"):
            with open(xtb_path / "charges", "w") as f:
                for charge in atomic_charges:
                    f.write(f"{charge}\n")
        run_xtb(
            elements,
            coordinates_,
            path=xtb_path,
            keywords=keywords,
            xcontrol_keywords=xcontrol_keywords,
        )
        with open(xtb_path / "gfnff_topo", "rb") as f:
            topology = f.read()
        topologies.append(topology)

    if paths is None:
        for temp_dir in temp_dirs:
            temp_dir.cleanup()

    return topologies


def setup_gfnff_calculators_python(
    elements: Union[Sequence[int], Sequence[str]],
    coordinates: Sequence[ArrayLike2D],
    charge: int = 0,
    solvent: Optional[str] = None,
) -> list[XTBCalculator]:
    """Sets up force fields for GFNFF calculation."""
    calculators = []
    for coordinates_ in coordinates:
        calculator = XTBCalculator(
            elements, coordinates_, charge=charge, solvent=solvent
        )
        _ = calculator.sp(return_gradient=False)
        calculators.append(calculator)
    return calculators


def opt_frags_from_complex(
    elements: Union[Sequence[int], Sequence[str]],
    coordinates: ArrayLike2D,
    keywords: Optional[list[str]] = None,
    wbo_keywords: Optional[list[str]] = None,
    xcontrol_keywords: Optional[MutableMapping[str, list[str]]] = None,
) -> list[tuple[Array1D, Array2D]]:
    """Optimize two fragments from complex.
    Args:
        elements: elements as symbols or numbers
        coordinates: coordinates [Å]
        keywords: xtb command line keywords for optimization
        wbo_keywords: xtb command line keywords for wbo calculation
        xcontrol_keywords: input instructions to write in the xtb xcontrol file
    Returns:
        fragments: Fragment elements and coordinates
    """
    elements = np.array(elements)
    coordinates = np.asarray(coordinates)
    bo_matrix = wbo_xtb(elements, coordinates, keywords=wbo_keywords)
    frag_indices = two_frags_from_bo(bo_matrix)
    fragments = []
    for indices in frag_indices:
        frag_elements = elements[indices]
        frag_coordinates = coordinates[indices]
        opt_coordinates = opt_xtb(
            frag_elements,
            frag_coordinates,
            keywords=keywords,
            xcontrol_keywords=xcontrol_keywords,
        )
        fragments.append((frag_elements, opt_coordinates))

    return fragments


def opt_constrained_complex(
    elements: Union[Sequence[int], Sequence[str]],
    coordinates: ArrayLike2D,
    distance_constraints: Optional[MutableMapping[tuple[int, int], float]] = None,
    atom_constraints: Optional[Sequence[int]] = None,
    fix_atoms: Optional[Sequence[int]] = None,
    keywords: Optional[list[str]] = None,
    xcontrol_keywords: Optional[MutableMapping[str, list[str]]] = None,
    fc: Optional[float] = None,
    path: Optional[Union[str, PathLike]] = None,
) -> Array2D:
    """Optimize constrained complex."""
    rmsd_atoms = set(range(1, len(elements) + 1))
    if distance_constraints is not None:
        if xcontrol_keywords is None:
            xcontrol_keywords = {}
        xcontrol_constraints = xcontrol_keywords.setdefault("constrain", [])
        if fc is not None:
            xcontrol_constraints.append(f"force constant={fc}")
        for (i, j), distance in distance_constraints.items():
            string = f"distance: {i}, {j}, {distance}"
            xcontrol_constraints.append(string)
            rmsd_atoms.difference_update({i, j})
        xcontrol_keywords["constrain"] = xcontrol_constraints
    if atom_constraints is not None:
        if xcontrol_keywords is None:
            xcontrol_keywords = {}
        xcontrol_atom_constraints = xcontrol_keywords.setdefault("constrain", [])
        atom_lines = textwrap.wrap(
            ", ".join(map(str, atom_constraints)), break_long_words=False
        )
        for line in atom_lines:
            fix_string = "atoms: " + line
            xcontrol_atom_constraints.append(fix_string)
        rmsd_atoms.difference_update(atom_constraints)
    if fix_atoms is not None:
        if xcontrol_keywords is None:
            xcontrol_keywords = {}
        xcontrol_fix_atoms = xcontrol_keywords.setdefault("fix", [])
        atom_lines = textwrap.wrap(
            ", ".join(map(str, fix_atoms)), break_long_words=False
        )
        for line in atom_lines:
            fix_string = "atoms: " + line
            xcontrol_fix_atoms.append(fix_string)
        rmsd_atoms.difference_update(fix_atoms)

    opt_coordinates = opt_xtb(
        elements,
        coordinates,
        keywords=keywords,
        xcontrol_keywords=xcontrol_keywords,
        path=path,
    )

    return opt_coordinates


def crest_constrained(
    elements: Union[Sequence[int], Sequence[str]],
    coordinates: ArrayLike2D,
    distance_constraints: Optional[MutableMapping[tuple[int, int], float]] = None,
    atom_constraints: Optional[Sequence[int]] = None,
    fix_atoms: Optional[Sequence[int]] = None,
    keywords: Optional[list[str]] = None,
    xcontrol_keywords: Optional[MutableMapping[str, list[str]]] = None,
    fc: Optional[float] = None,
    path: Optional[Union[str, PathLike]] = None,
) -> ConformerEnsemble:
    """Run constrained CREST calculation."""
    rmsd_atoms = set(range(1, len(elements) + 1))
    if distance_constraints is not None:
        if xcontrol_keywords is None:
            xcontrol_keywords = {}
        xcontrol_constraints = xcontrol_keywords.setdefault("constrain", [])
        if fc is not None:
            xcontrol_constraints.append(f"force constant={fc}")
        for (i, j), distance in distance_constraints.items():
            string = f"distance: {i}, {j}, {distance}"
            xcontrol_constraints.append(string)
            rmsd_atoms.difference_update({i, j})
        xcontrol_keywords["constrain"] = xcontrol_constraints
    if atom_constraints is not None:
        if xcontrol_keywords is None:
            xcontrol_keywords = {}
        xcontrol_atom_constraints = xcontrol_keywords.setdefault("constrain", [])
        atom_lines = textwrap.wrap(
            ", ".join(map(str, atom_constraints)), break_long_words=False
        )
        for line in atom_lines:
            fix_string = "atoms: " + line
            xcontrol_atom_constraints.append(fix_string)
        rmsd_atoms.difference_update(atom_constraints)
    if fix_atoms is not None:
        if xcontrol_keywords is None:
            xcontrol_keywords = {}
        xcontrol_fix_atoms = xcontrol_keywords.setdefault("fix", [])
        atom_lines = textwrap.wrap(
            ", ".join(map(str, fix_atoms)), break_long_words=False
        )
        for line in atom_lines:
            fix_string = "atoms: " + line
            xcontrol_fix_atoms.append(fix_string)
        rmsd_atoms.difference_update(fix_atoms)
    if len(rmsd_atoms) > 0:
        if xcontrol_keywords is None:
            xcontrol_keywords = {}
        xcontrol_rmsd_atoms = xcontrol_keywords.setdefault("metadyn", [])
        atom_lines = textwrap.wrap(
            ", ".join(map(str, rmsd_atoms)), break_long_words=False
        )
        for line in atom_lines:
            fix_string = "atoms: " + line
            xcontrol_rmsd_atoms.append(fix_string)

    conformer_ensemble = opt_crest(
        elements,
        coordinates,
        keywords=keywords,
        xcontrol_keywords=xcontrol_keywords,
        path=path,
    )

    return conformer_ensemble


def calculate_e_shift_xtb(
    elements: Union[Sequence[int], Sequence[str]],
    coordinates: Sequence[ArrayLike2D],
    topologies: Sequence[bytes],
    e_diff_ref: Optional[float] = None,
    keywords_ff: Optional[list[str]] = None,
    keywords_sp: Optional[list[str]] = None,
    xcontrol_keywords_ff: Optional[MutableMapping[str, list[str]]] = None,
    xcontrol_keywords_sp: Optional[MutableMapping[str, list[str]]] = None,
    paths: Optional[Sequence[Union[str, PathLike]]] = None,
) -> tuple[float, float, float]:
    """Calculate energy shift between reference (default: GFN2-xTB) and GFN-FF reaction energies.
    Args:
        elements: elements as symbols or numbers
        coordinates: sequence containing the coordinates of each ground states [Å]
        topologies: sequence of GFN-FF topologies for each ground state
        e_diff_ref: reference reaction energy [Eh]. If provided, it is used instead of the GFN2-xTB calculation
        keywords_ff: xtb command line keywords for GFN-FF calculation
        keywords_sp: xtb command line keywords for GFN2-xTB calculation
        xcontrol_keywords_ff: input instructions to write in the xtb xcontrol file for GFN-FF calculation
        xcontrol_keywords_sp: input instructions to write in the xtb xcontrol file for GFN2-xTB calculation
        paths: list of folders to save the xtb runs
    Returns:
        e_shift: difference between the GFN2-xTB and GFN-FF reaction energies
        e_diff_ref: reaction energy calculated with GFN2-xTB or given as argument
        e_diff_ff: reaction energy calculated with GFN-FF
    """

    # Set the xtb keywords for the GFN-FF calculations
    if keywords_ff is None:
        keywords_ff = []
    keywords_ff = set([keyword.strip().lower() for keyword in keywords_ff])
    # Give the --gfnff keyword (--gfn2 by default)
    keywords_ff.add("--gfnff")

    if paths is None:
        temp_dirs = [
            TemporaryDirectory(dir=config.TMP_DIR) for i in range(len(coordinates))
        ]
        xtb_paths = [Path(temp_dir.name) for temp_dir in temp_dirs]
    else:
        xtb_paths = [Path(path) for path in paths]

    energies_ff = []
    if not e_diff_ref:
        energies_sp = []
    for coordinates_, topology, xtb_path in zip(coordinates, topologies, xtb_paths):
        xtb_path.mkdir(exist_ok=True)
        with open(xtb_path / "gfnff_topo", "wb") as f:
            f.write(topology)
        run_xtb(
            elements,
            coordinates_,
            path=xtb_path,
            keywords=keywords_ff,
            xcontrol_keywords=xcontrol_keywords_ff,
        )
        energy = parse_energy(xtb_path / "xtb.out")
        energies_ff.append(energy)

        if not e_diff_ref:
            run_xtb(
                elements,
                coordinates_,
                path=xtb_path,
                keywords=keywords_sp,
                xcontrol_keywords=xcontrol_keywords_sp,
            )
            energy = parse_energy(xtb_path / "xtb.out")
            energies_sp.append(energy)

    if paths is None:
        for temp_dir in temp_dirs:
            temp_dir.cleanup()

    if not e_diff_ref:
        e_diff_ref = energies_sp[-1] - energies_sp[0]
    e_diff_ff = energies_ff[-1] - energies_ff[0]
    e_shift = e_diff_ref - e_diff_ff

    return e_shift, e_diff_ref, e_diff_ff


def calculate_e_shift_xtb_python(
    calculators: Sequence[XTBCalculator], method: str = ("GFN2-xTB")
) -> ShiftResults:
    """Calculate energy shift between reference (default: GFN2-xTB) and GFN-FF reaction energies."""
    energies_gfn = []
    energies_ff = []
    for calculator in calculators:
        energy_ff = calculator.sp(return_gradient=False)
        calculator_sp = XTBCalculator(
            calculator.elements,
            calculator.coordinates,
            method=method,
            charge=calculator.charge,
            solvent=calculator.solvent,
        )
        energy_gfn = calculator_sp.sp(return_gradient=False)
        energies_gfn.append(energy_gfn)
        energies_ff.append(energy_ff)
    energy_diff_gfn = energies_gfn[-1] - energies_gfn[0]
    energy_diff_ff = energies_ff[-1] - energies_ff[0]
    energy_shift = energy_diff_gfn - energy_diff_ff

    results = ShiftResults(
        energy_shift=energy_shift,
        energy_diff_gfn=energy_diff_gfn,
        energy_diff_ff=energy_diff_ff,
        energies_gfn=energies_gfn,
        energies_ff=energies_ff,
    )

    return results
