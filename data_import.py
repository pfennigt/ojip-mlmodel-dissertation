
import pandas as pd
import numpy as np
from pandas.core.indexing import _IndexSlice
import matplotlib.pyplot as plt
from pandas.core.dtypes.common import is_numeric_dtype

from pathlib import Path

idx: _IndexSlice = pd.IndexSlice

# %%
# Define the path to the data
DATA_PATH = Path("data/01_known_effects")

# %%
# Load the list of samples
samples = pd.read_csv(DATA_PATH / "experiments.csv", index_col=0, decimal=",")

# Create an index from the experiment table
samples_index_fields = {
    "Label": "label",
    "Effect in PSET": "effect",
    "Treatment": "treatment",
    "Experimenter, location": "experimenter_location",
    "Strain": "strain",
    "CO2 level": "CO2_level",
    "Cultivation + experiment temperature": "temperature",
    "Cultivation light intensity": "Light_intensity",
    "Dark or light acclimated": "light_acclimation",
    "Growth light color (nm)": "light_color",
    "Cultivator": "cultivator",
    "Medium": "medium",
    "Fluorometer": "fluorometer",
    "SP color (nm)": "SP_color",
    "SP intensity": "SP_intensity",
    "Measuring vessel": "vessel",
    "OD680 MC-1000": "OD680",
    "OD720 MC-1000": "OD720",
    "ΔOD": "deltaOD",
    "OD680/720 raw": "OD680_720",
}
samples_index_fields_inv = {v: k for k, v in samples_index_fields.items()}

samples_index = pd.MultiIndex.from_frame(
    samples.reset_index().loc[:, list(samples_index_fields.keys())]
)


# %%
def get_index_levels(df, level):
    columns = df.columns.remove_unused_levels()

    # Get the levels and the index corresponding to the level
    _level = samples_index_fields_inv[level]
    level_ind = np.where(np.array(list(df.columns.names)) == _level)[0][0]
    return list(columns.levels[level_ind])


# %%
# Summary statistics
fig = plt.figure(constrained_layout=True, figsize=(15, 15))
gs = fig.add_gridspec(2,2, height_ratios=[1,2])

sfig1 = fig.add_subfigure(gs[:,0])
sfig2 = fig.add_subfigure(gs[0,1])
sfig3 = fig.add_subfigure(gs[1,1])

axes = sfig1.subplots(int(np.ceil(len(samples.columns[2:]) / 3)), 3)

for column, ax in zip(samples.columns[2:], axes.flatten()):
    # Get the values dor each column and count the occurrences
    dat = samples.loc[:, column]
    if not is_numeric_dtype(dat.dtype):
        dat = dat.value_counts()
        dat.plot(kind="bar", ax=ax)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=10, ha="right")
    else:
        dat.plot(kind="hist", ax=ax)

    ax.set_title(column, size=10)
    ax.set_ylabel("")
    ax.set_xlabel("")

fig.subplots_adjust(wspace=0.3, hspace=0.7)

for column, sfig in zip(samples.columns[:2], [sfig3, sfig2]):
    ax = sfig.subplots()
    dat = samples.loc[:, column]
    dat = dat.value_counts()
    dat.plot(kind="bar", ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.set_title(column, size=10)

# %%
# for i in samples_index_fields.values():
#     print(f"{i},")# : str | list[str | int] | slice | None = slice(None),")


# Create a function for easy indexing
def didx(
    label: str | list[str | int] | slice | None = slice(None),
    effect: str | list[str | int] | slice | None = slice(None),
    treatment: str | list[str | int] | slice | None = slice(None),
    experimenter_location: str | list[str | int] | slice | None = slice(None),
    strain: str | list[str | int] | slice | None = slice(None),
    CO2_level: str | list[str | int] | slice | None = slice(None),
    temperature: str | list[str | int] | slice | None = slice(None),
    Light_intensity: str | list[str | int] | slice | None = slice(None),
    light_acclimation: str | list[str | int] | slice | None = slice(None),
    light_color: str | list[str | int] | slice | None = slice(None),
    cultivator: str | list[str | int] | slice | None = slice(None),
    medium: str | list[str | int] | slice | None = slice(None),
    fluorometer: str | list[str | int] | slice | None = slice(None),
    SP_color: str | list[str | int] | slice | None = slice(None),
    SP_intensity: str | list[str | int] | slice | None = slice(None),
    vessel: str | list[str | int] | slice | None = slice(None),
    OD680: str | list[str | int] | slice | None = slice(None),
    OD720: str | list[str | int] | slice | None = slice(None),
    deltaOD: str | list[str | int] | slice | None = slice(None),
    OD680_720: str | list[str | int] | slice | None = slice(None),
) -> int | str | slice:
    res: list[str | list[str | int] | slice | None] = [
        x
        for x in [
            label,
            effect,
            treatment,
            experimenter_location,
            strain,
            CO2_level,
            temperature,
            Light_intensity,
            light_acclimation,
            light_color,
            cultivator,
            medium,
            fluorometer,
            SP_color,
            SP_intensity,
            vessel,
            OD680,
            OD720,
            deltaOD,
            OD680_720,
        ]
        if x is not None
    ]
    return idx[*res]


# %% [markdown]
# ## Load the data

# %%
# Get the paths to the samples files
files = {}

for i in samples_index.get_level_values(0):
    _index = f"{i:04}"

    # Get the file to the current index
    file = list(DATA_PATH.glob(f"ojip_data/{_index}*"))[0]

    # Set the options for reading the data based on the used fluorometer
    if samples.loc[i, "Fluorometer"] == "MULTI-COLOR-PAM":
        skiprows = 0
        skipfooter = 0
        sep = ";"
        index_col = 0
        select_col = "Fluo, V"
        time_to_ms = 1
    elif samples.loc[i, "Fluorometer"] == "AquaPen":
        skiprows = 7
        skipfooter = 38
        sep = r"\s"
        index_col = 0
        select_col = "OJIP"
        time_to_ms = 1e-3
    else:
        print(i, samples.loc[i, "Fluorometer"])
        break

    # Read the data with the pre-defined options
    _df = pd.read_table(
        file,
        skiprows=skiprows,
        skipfooter=skipfooter,
        sep=sep,
        index_col=index_col,
        engine="c" if skipfooter == 0 else "python",
    )[select_col]

    _df.index = pd.Index(np.round(_df.index * time_to_ms, 2))

    # Save the data
    files[i] = _df

# Concatenate the data
df = pd.DataFrame(files).sort_index(axis=1)
df.columns = samples_index

# %%
# Get scaling factors from the fluorometer comparison data
SCALING_DATA_PATH = Path("data/00_fluorometer_data_scaling")

scaling_samples = pd.read_csv(
    SCALING_DATA_PATH / "experiments.csv", index_col=0, decimal=","
)

# Get the paths to the samples files
files = {}

for i in scaling_samples.index:
    _index = f"{i:02}"

    # Get the file to the current index
    file = list(SCALING_DATA_PATH.glob(f"ojip_data/{_index}*"))[0]

    # Set the options for reading the data based on the used fluorometer
    if scaling_samples.loc[i, "Fluorometer"] == "MULTI-COLOR-PAM":
        skiprows = 0
        skipfooter = 0
        sep = ";"
        index_col = 0
        select_col = "Fluo, V"
        time_to_ms = 1
    elif scaling_samples.loc[i, "Fluorometer"] == "AquaPen":
        skiprows = 7
        skipfooter = 38
        sep = r"\s"
        index_col = 0
        select_col = "OJIP"
        time_to_ms = 1e-3
    else:
        print(i, scaling_samples.loc[i, "Fluorometer"])
        break

    # Read the data with the pre-defined options
    _df = pd.read_table(
        file,
        skiprows=skiprows,
        skipfooter=skipfooter,
        sep=sep,
        index_col=index_col,
        engine="c" if skipfooter == 0 else "python",
    )[select_col]

    _df.index = pd.Index(np.round(_df.index * time_to_ms, 2))

    # Save the data
    files[i] = _df

# Concatenate the data
scaling_df = pd.DataFrame(files).sort_index(axis=1)
scaling_df.columns = pd.MultiIndex.from_frame(scaling_samples)

# %%
# Determine scaling factors between devices
factors = pd.DataFrame(
    index=pd.MultiIndex(
        [[], [], [], []],
        [[], [], [], []],
        name=["from", "from SP (nm)", "to", "to SP (nm)"],
    )
)

for hue in scaling_df.columns.levels[-1]:
    _dat = scaling_df.loc[:, idx[:, :, :, hue]].loc[0:0.05].mean().droplevel([0, -1])
    MCPAM = _dat.loc[idx[["MULTI-COLOR-PAM"], :]]
    AQPEN = _dat.loc[idx[["AquaPen"], :]]

    factors.loc[idx[*AQPEN.index[0], *MCPAM.index[0]], "F0"] = (
        MCPAM.iloc[0] / AQPEN.iloc[0]
    )

# factors["Fm"] = (
#     scaling_df.loc[:, idx[:, "MULTI-COLOR-PAM",:]].max().droplevel([0,1,2])
#     / scaling_df.loc[:, idx[:, "AquaPen",:]].max().droplevel([0,1,2])
# )

# factors = pd.DataFrame(factors)

# %%
# Determine scaling factors between devices at F0 and FM
factors = pd.DataFrame(
    index=pd.MultiIndex(
        [[], [], [], []],
        [[], [], [], []],
        name=["from", "from SP (nm)", "to", "to SP (nm)"],
    )
)

for hue in scaling_df.columns.levels[-1]:
    MCPAM = scaling_df.loc[:, idx[:, "MULTI-COLOR-PAM", :, hue]].droplevel(
        [0, -1], axis=1
    )
    AQPEN = scaling_df.loc[:, idx[:, "AquaPen", :, hue]].droplevel([0, -1], axis=1)

    # F0
    factors.loc[idx[*AQPEN.columns[0], *MCPAM.columns[0]], "F0"] = (
        MCPAM.loc[0:0.05].mean().iloc[0] / AQPEN.loc[0:0.05].mean().iloc[0]
    )

    # FM
    factors.loc[idx[*AQPEN.columns[0], *MCPAM.columns[0]], "FM"] = (
        MCPAM.max().iloc[0] / AQPEN.max().iloc[0]
    )

factors["mean"] = factors.mean(axis=1)

# %% [markdown]
# ### Rescale AquaPen samples

# %%
# Rescale AquaPen samples with the determined conversion factors
for (fluorometer, SP_color, _, _), factor in factors["mean"].items():
    try:
        df.loc[:, didx(fluorometer=fluorometer, SP_color=SP_color)] = (
            df.loc[:, didx(fluorometer=fluorometer, SP_color=SP_color)] * factor
        )
    except KeyError:
        pass
