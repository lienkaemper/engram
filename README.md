# Engram

Spiking model for "CA1 Engram Cell Dynamics Before and After Learning": https://www.biorxiv.org/content/10.1101/2024.04.16.589790v1.abstract

To make Figure 5B:

```
python sim_low_inhib.py
python theory_low_inhib.py
python plot_low_inhib.py
```

To make Figure 5C (lineplot):

```
python theory_specific_inhib.py
python plot_specific_inhib.py
```

To make the four barpliots in Figure 5 C, D, F:

``` 
python sim_vary_inhib.py
python theory_vary_inhib.py
python plot_barplots.py
```

To make Figure 5E: 

```
python sim_vary_inhib.py
python theory_vary_inhib.py
python plot_correlation_ratio.py
```

To make Figure 5H:

```
python sim_low_inhib.py
python plot_noisy_tagging_heatmaps.py
```

To make Figure 6 C, D, E:

```
python reactivation.py
```

To make Supp Figure 6 A:
```
python sim_low_inhib_with_excitability.py
python theory_low_inhib_with_excitability.py
python plot_low_inhib_with_excitability.py
```

To make Supp Figure 6B:

```
python theory_vary_inhib_levels.py
python plot_decomposition.py
```
