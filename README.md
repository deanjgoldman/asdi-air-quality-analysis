# ASDI AQ View
This repository supports a pipeline for:
1. Downloading and merging two datasets
- SILAM Air Quality Dataset provided by: Finnish Meteorological Institute [1]
- High Resolution Population Density Maps + Demographic Estimates by CIESIN and Meta [2]

2. Conducting analysis and creating visualizations to answer questions such as:
- How many people are estimated to be affected by an unhealthy level of pollution, for any given pollutant?
- Where are the sources of pollution? Do they correlate generally with population density?
- For a given set of countries, how does pollution correlate with population density? Do high-pollution areas tend to exist outside of population centers?

# ETL
1. Install the dependencies listed in `requirements.txt`.
2. Configure input/output directory settings in `settings.py`
3. Run `python etl.py` to download the SILAM and CIESIN/Meta datasets and map them onto the same coordinate system. This may take several minutes.
4. Run analysis/visualization tools inside of `/scripts`.

# Analysis
The following scripts can support analysis of certain questions pertinent to how pollution affects populations, where pollution occurs, and what is the characteristic relatonship between population and pollution for a given set of countries?

### How many people live with a certain amount of pollution? 
An example output from `query.py`. This script produces a cumulative frequency distribution for units of population with a pollution level greater than or a certain amount. Drawn on the x-axis is the level determined by either EPA or CDC to be unhealthy [3][4]. Note: These thresholds were picked since they were available publicly, but custom thresholds could easily be swapped in.

![cdf PM10](/output/cfd_pm10.png)  

## What countries have similar or different pollution/population density profiles?
An example output from `compare_histograms_similar_scale.py` produces a grid of heatmaps, all in the same scale space. These heatmaps represent the frequency of population density and pollution concentration with a customizable binning arrangement. User can determine the countries and pollutants to graph. 

![hist compare US SAU THAI](/output/hist_compare_US_SAU_THAI.png)  

Hypothetically, a user may look to distinguish countries with different energy mixes (e.g. France's energy mix is very different from UAE's). See https://ourworldindata.org/energy-mix#.

To produce this heatmap, but to individuate each country by it's own scale, `compare_histograms_individual_scale.py` will produce the same heatmap grid as above, but each country's population/pollution axis is binned into equally spaced bins determined by variabls: `pol_bins` and `pop_bins`.

![hist compare ind scale US SAU THAI](/output/hist_compare_scale_unique_US_SAU_THAI.png)  

## What are the sources of a particular pollutant?
Some pollutants like NO2/O3 are known to originate from burning coal, oil, and diesel, while PM/CO is associated more with dust, agricultural pollution, or fires. `plot_divergence_population.py` will produce a world map, with two heatmaps overlayed reperesenting the divergence values in pollution concentration for a particular pollutant, along with the population density estimates. These values are aggregated over the entire dataset. A sequence of daily pollution concentrations can be produced with `plot_divergence_daily.py`.

Example: plotting O3 pollution vs PM10 pollution between Middle East/Africa and the Northern/Central Americas

![PM10 pollution Middle East/Africa](/output/pollution_PM10_middle_east_africa.png)  

![PM10 pollution North/Central Americas](/output/pollution_PM10_north_central_america.png)

![O3 pollution Middle East/Africa](/output/pollution_O3_middle_east_africa.png)  

![O3 pollution North/Central Americas](/output/pollution_O3_north_central_america.png)

# Remarks
This repository can be used as a toolset for aggregating and analyzing these two data sources. Some fruitful next steps might be to incorporate energy mix profiles by country into this analysis, in order to estimate the distribution of the pollution source across energy production methods. With this, an analyst could produce some estimate of the reduction in pollution given some shift in the distribution of energy production methods, i.e. if majority coal producing countries shifted 10% of their total energy mix to nuclear, how much aggregate SO2 and NO pollution would decrease?

# References
[1] SILAM Air Quality was accessed on July 27, 2022 from https://registry.opendata.aws/silam.  
  
[2] High Resolution Population Density Maps + Demographic Estimates by CIESIN and Meta was accessed on July 27, 2022 from https://registry.opendata.aws/dataforgood-fb-hrsl. Meta and Center for International Earth Science Information Network - CIESIN - Columbia University. 2022. High Resolution Settlement Layer (HRSL). Source imagery for HRSL © 2016 Maxar. Accessed 27 07 2022.  
  
[3] United States Environmental Protection Agency (EPA) https://www.epa.gov/topics-epa-web  
  
[4] The National Institute for Occupational Safety and Health (NIOSH). Centers for Disease Control and Prevention (CDC). https://www.cdc.gov/niosh/   



